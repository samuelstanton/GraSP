import torch
import torch.nn as nn
import torch.nn.functional as F
from grasp.utils.prune_utils import get_gradients
from grasp.models.base.graph_utils import GraphEdge, GraphLayer
from grasp.utils.common_utils import try_cuda
from collections import OrderedDict
from grasp.pruner.weight_mag import weight_mag_pruner
import copy

from tqdm import tqdm


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    inputs_by_class = [torch.cat(inputs) for inputs in datas]
    targets_by_class = [torch.cat(targets) for targets in labels]
    return inputs_by_class, targets_by_class


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def get_grad_norm(net, dataset, norm_factor=1.):
    module_types = (nn.Conv2d, nn.Linear, GraphEdge)
    weights = []
    for m in net.modules():
        if isinstance(m, module_types):
            weights.append(m.weight)
    grads = [torch.zeros_like(w) for w in weights]
    net.zero_grad()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    for inputs, targets in tqdm(dataloader):
        inputs, targets = try_cuda(inputs), try_cuda(targets)
        logits = net(inputs)
        loss = nn.functional.cross_entropy(logits, targets, reduction='sum') / len(dataset)
        batch_grads = get_gradients(loss, weights, allow_unused=True, create_graph=False)
        for i, batch_grad in enumerate(batch_grads):
            grads[i] += batch_grad
    abs_grad_norm = torch.stack([grad.pow(2).mean() for grad in grads]).mean().sqrt()
    rel_grad_norm = (abs_grad_norm / norm_factor).item()
    return rel_grad_norm


def correct_grads(modules, weights, grads):
    corrected_grads = copy.deepcopy([g.detach() for g in grads])
    for i, (m, w, grad) in enumerate(zip(modules, weights, grads)):
        if isinstance(m, GraphEdge):
            corrected_grads[i] = grad / (w.exp() ** 2)
    return corrected_grads


def correct_hv_prods(modules, weights, grads, hv_prods):
    corrected_hv_prods = copy.deepcopy([hv.detach() for hv in hv_prods])
    for i, (m, w, g, hv) in enumerate(zip(modules, weights, grads, hv_prods)):
        if isinstance(m, GraphEdge):
            corrected_hv_prods[i] = (hv - g ** 2 / w.exp() ** 2) / w.exp()
    return corrected_hv_prods


def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True,
          mode='prune_weights'):
    eps = 1e-10
    keep_ratio = 1-ratio
    net.zero_grad()

    layer_types = (nn.Conv2d, nn.Linear)
    prunable_modules, weights = [], []
    for layer in net.modules():
        if isinstance(layer, GraphLayer) and mode == 'prune_ops':
            for edge in layer.edges:
                weights.append(edge.weight)
                prunable_modules.append(edge)
        elif isinstance(layer, layer_types) and mode == 'prune_weights':
            weights.append(layer.weight)
            prunable_modules.append(layer)

    for w in weights:
        w.requires_grad_(True)

    inputs_by_class, targets_by_class = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
    hess_grad_prods = [torch.zeros_like(w) for w in weights]
    print("==== computing gradients and Hessian-grad vector products ====")
    for inputs, targets in zip(inputs_by_class, targets_by_class):
        inputs, targets = inputs.to(device), targets.to(device)
        scaled_logits = net(inputs) / T
        loss = F.cross_entropy(scaled_logits, targets, reduction='sum') / (num_classes * samples_per_class)
        grads = get_gradients(loss, weights, allow_unused=True, create_graph=True)
        corrected_grads = correct_grads(prunable_modules, weights, grads)
        grad_inner_prod = torch.stack(
            [(grad_1 * grad_2.detach()).sum() for grad_1, grad_2 in zip(grads, corrected_grads)]
        ).sum()
        class_hg_prods = get_gradients(grad_inner_prod, weights, allow_unused=True, create_graph=False)
        class_hg_prods = correct_hv_prods(prunable_modules, weights, grads, class_hg_prods)
        for i, hg_prod in enumerate(class_hg_prods):
            hess_grad_prods[i] += hg_prod

    print("==== computing weight scores ====")
    scores = OrderedDict()
    for module, weight, hg_prod in zip(prunable_modules, weights, hess_grad_prods):
        if isinstance(module, GraphEdge):
            scores[module] = -weight.exp() * hg_prod
            import pdb;
            pdb.set_trace()
        else:
            scores[module] = -weight * hg_prod


    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    print("==== masking weights ====")
    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    keep_masks = OrderedDict()

    # original GraSP masking rule
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    acceptable_score = threshold[-1]
    for m, g in scores.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    # GraSP-Magnitude scoring rule
    # threshold, _ = torch.topk(-all_scores.abs(), num_params_to_rm, sorted=True)
    # acceptable_score = -threshold[-1]
    # for m, score in scores.items():
    #     keep_masks[m] = ((score / norm_factor).abs() >= acceptable_score).float()
    print('** accept: ', acceptable_score)

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks


def prune_weights(config, mb, trainloader, num_classes):
    ratio = config.target_weight_ratio
    assert ratio > 0.
    if config.pruner.lower() == 'grasp':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        masks = GraSP(mb.model, ratio, trainloader, device,
                      num_classes=num_classes,
                      samples_per_class=config.samples_per_class,
                      num_iters=config.get('num_iters', 1),
                      mode='prune_weights')
    elif config.pruner.lower() == 'weight_mag':
        masks = weight_mag_pruner(mb.model, ratio, mode='prune_weights')
    else:
        raise RuntimeError('unsupported weight pruner')

    mb.register_mask(masks)
    return mb
