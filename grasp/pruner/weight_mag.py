import torch
from torch import nn
from grasp.models.base.graph_utils import GraphLayer
from collections import OrderedDict


def weight_mag_pruner(net, ratio, mode='prune_weights'):
    eps = 1e-10
    keep_ratio = 1 - ratio
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

    print("==== computing weight scores ====")
    scores = OrderedDict()
    for module, weight in zip(prunable_modules, weights):
        scores[module] = weight.abs()
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = all_scores.max() + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    print("==== masking weights ====")
    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    keep_masks = OrderedDict()

    threshold, _ = torch.topk(-all_scores, num_params_to_rm, sorted=True)
    acceptable_score = -threshold[-1]
    for m, score in scores.items():
        keep_masks[m] = ((score / norm_factor) >= acceptable_score).float()
    print('** accept: ', acceptable_score)

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks