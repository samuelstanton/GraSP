import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from grasp.utils.prune_utils import get_gradients
from grasp.models.base.graph_utils import GraphEdge

import copy


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


def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True,
          mode='prune_weights'):
    eps = 1e-10
    keep_ratio = 1-ratio
    net.zero_grad()

    module_types = (GraphEdge) if mode == 'prune_ops' else (nn.Conv2d, nn.Linear)

    prunable_modules, weights = [], []
    for layer in net.modules():
        if isinstance(layer, module_types):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad_(True)
            weights.append(layer.weight)
            prunable_modules.append(layer)

    inputs_by_class, targets_by_class = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
    hess_grad_prods = [torch.zeros_like(w) for w in weights]
    print("==== computing gradients and Hessian-grad vector products ====")
    for inputs, targets in zip(inputs_by_class, targets_by_class):
        inputs, targets = inputs.to(device), targets.to(device)
        scaled_logits = net(inputs) / T
        loss = F.cross_entropy(scaled_logits, targets, reduction='sum') / (num_classes * samples_per_class)
        grads = get_gradients(loss, weights, allow_unused=True, create_graph=True)
        grad_inner_prod = torch.stack(
            [(grad_1 * grad_2.detach()).sum() for grad_1, grad_2 in zip(grads, grads)]
        ).sum()
        class_hg_prods = get_gradients(grad_inner_prod, weights, allow_unused=True, create_graph=False)
        for i, hg_prod in enumerate(class_hg_prods):
            hess_grad_prods[i] += hg_prod

    print("==== computing weight scores ====")
    scores = dict()
    for module, weight, hg_prod in zip(prunable_modules, weights, hess_grad_prods):
        scores[module] = -weight * hg_prod
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    print("==== masking weights ====")
    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in scores.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks


def prune_weights(config, mb, trainloader, num_classes):
    num_iterations = config.iterations
    target_ratio = config.target_weight_ratio
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    # ====================================== start pruning ======================================
    for iteration in range(num_iterations):
        print("Iteration of: %d/%d" % (iteration + 1, num_iterations))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        masks = GraSP(mb.model, ratio, trainloader, device,
                      num_classes=num_classes,
                      samples_per_class=config.samples_per_class,
                      num_iters=config.get('num_iters', 1))
        print('=> Using GraSP')
        # ========== register mask ==================
        mb.register_mask(masks)
    return mb
