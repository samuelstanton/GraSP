import torch

from grasp.pruners.grasp import GraSP
from grasp.pruners.weight_mag import weight_mag_pruner


def prune_weights(config, mb, trainloader):
    pruner_type = config.weight_pruner.type.lower()
    ratio = config.weight_pruner.target_percent / 100
    assert ratio > 0.
    if 'grasp' in pruner_type:
        _, rank_by = pruner_type.split('_')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        masks = GraSP(mb.model, ratio, trainloader, device,
                      num_classes=config.dataset.num_classes,
                      samples_per_class=config.samples_per_class,
                      logit_scale=config.grasp_logit_scale,
                      mode='prune_weights',
                      rank_by=rank_by)
    elif pruner_type == 'weight_mag':
        masks = weight_mag_pruner(mb.model, ratio, mode='prune_weights')
    else:
        raise RuntimeError('unsupported weight pruner')

    mb.register_mask(masks)
    return mb
