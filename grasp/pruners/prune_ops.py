import os
import torch

from grasp.models.graphnet import get_rand_op_masks
from grasp.utils.graph_utils import GraphLayer
from grasp.pruners.grasp import GraSP
from grasp.pruners.weight_mag import weight_mag_pruner


def prune_ops(config, mb, trainloader):
    pruner_type = config.op_pruner.type.lower()
    num_iterations = config.op_pruner.num_iter
    target_ratio = config.op_pruner.target_percent / 100
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    # ====================================== start pruning ======================================
    for iteration in range(num_iterations):
        if 'grasp' in pruner_type:
            _, rank_by = pruner_type.split('_')
            print("GraSP iteration: %d/%d" % (iteration + 1, num_iterations))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            masks = GraSP(mb.model, ratio, trainloader, device,
                          num_classes=config.dataset.num_classes,
                          samples_per_class=config.op_pruner.samples_per_class,
                          logit_scale=config.grasp_logit_scale,
                          mode='prune_ops',
                          rank_by=rank_by)

        elif pruner_type == 'weight_mag':
            masks = weight_mag_pruner(mb.model, ratio, mode='prune_ops')

        elif pruner_type == 'random':
            masks = get_rand_op_masks(mb.model, ratio)

        else:
            raise RuntimeError("unrecognized operation pruner")

        for layer in mb.model.modules():
            if isinstance(layer, GraphLayer):
                edge_masks = [masks.popitem(last=False)[1].bool() for _ in layer.edges]
                layer.sparsify(edge_masks)

    stage_count = 0
    for layer in mb.model.modules():
        if isinstance(layer, GraphLayer):
            layer.draw_graph(os.getcwd(), f"stage_{stage_count}_graph", view=config.view_graphs)
            stage_count += 1
    return mb