import hydra
import pandas as pd
from upcycle.random.seed import set_all_seeds
import random
import torch

from grasp.utils.data_utils import get_dataloader
from grasp.utils.network_utils import get_grad_norm
from grasp.models.model_base import get_model_base
from grasp import pruners
from grasp.boilerplate import training_loop
from grasp.models import GraphNet
from omegaconf import OmegaConf, DictConfig


def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    print(hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")

    return hydra_cfg, logger


@hydra.main(config_path='../hydra/main.yaml')
def main(config):
    # construct logger, model, dataloaders
    config, s3_logger = startup(config)
    model_base = get_model_base(config)
    trainloader, testloader = get_dataloader(dataset_name=config.dataset.name, **config.dataset)

    print("==== computing initial metrics ====")
    s3_logger.add_table('pruning_metrics')
    abs_grad_norm = 1. if config.debug is True else get_grad_norm(model_base.model, trainloader.dataset)
    rel_grad_norm = 1.
    s3_logger.log(dict(num_params=model_base.num_params, num_ops=model_base.num_ops, grad_norm=rel_grad_norm),
                  'init', 'pruning_metrics')

    print("==== pruning ops ====")
    if config.op_pruner.target_percent > 0. and isinstance(model_base.model, GraphNet):
        model_base = pruners.prune_ops(config, model_base, trainloader)
        rel_grad_norm = 1. if config.debug else get_grad_norm(model_base.model, trainloader.dataset,
                                                              norm_factor=abs_grad_norm)
    s3_logger.log(dict(num_params=model_base.num_params, num_ops=model_base.num_ops, grad_norm=rel_grad_norm),
                  'prune_ops', 'pruning_metrics')

    print("==== pruning weights ====")
    if config.weight_pruner.target_percent > 0.:
        model_base = pruners.prune_weights(config, model_base, trainloader)
        rel_grad_norm = 1. if config.debug else get_grad_norm(model_base.model, trainloader.dataset,
                                                              norm_factor=abs_grad_norm)
    prune_metrics = model_base.get_ratio_at_each_layer()
    s3_logger.log(dict(num_params=prune_metrics['remaining_params'], num_ops=model_base.num_ops, grad_norm=rel_grad_norm),
                  'prune_weights', 'pruning_metrics')

    print(pd.DataFrame(s3_logger.data['pruning_metrics']).to_markdown())
    s3_logger.write_csv()

    print("==== training the network ====")
    training_loop(
        net=model_base.model,
        trainloader=trainloader,
        testloader=testloader,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        num_epochs=config.num_train_epochs,
        s3_logger=s3_logger
    )


if __name__ == '__main__':
    main()
