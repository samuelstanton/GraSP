import torch

from grasp.pruners.grasp import GraSP
from grasp.pruners.weight_mag import weight_mag_pruner


def prune_weights(config, mb, trainloader):
    pruner_type = config.weight_pruner.type.lower()
    ratio = config.target_weight_ratio
    assert ratio > 0.
    if pruner_type == 'grasp':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        masks = GraSP(mb.model, ratio, trainloader, device,
                      num_classes=config.dataset.num_classes,
                      samples_per_class=config.samples_per_class,
                      num_iters=1,
                      mode='prune_weights')
    elif pruner_type == 'weight_mag':
        masks = weight_mag_pruner(mb.model, ratio, mode='prune_weights')
    else:
        raise RuntimeError('unsupported weight pruner')

    mb.register_mask(masks)
    return mb
