import os
import hydra
import pandas as pd
from upcycle.logging import S3Logger
from upcycle.random.seed import set_all_seeds
import random
import torch

from grasp.models import ModelBase
from grasp.models.base.init_utils import weights_init
from grasp.utils.common_utils import (try_cuda)
from grasp.utils.data_utils import get_dataloader
from grasp.utils.network_utils import get_network
from grasp.pruner.GraphGraSP import prune_ops
from grasp.pruner.GraSP import prune_weights
from grasp.boilerplate import training_loop


def init_s3_logger(config):
    data_dir = os.path.join(config.s3_path, config.exp_name)
    s3_logger = S3Logger(data_dir, config.s3_bucket)
    return s3_logger


@hydra.main(config_path='../hydra/main.yaml')
def main(config):
    if config.seed is None:
        seed = random.randint(0, 100000)
        config['seed'] = seed
        set_all_seeds(seed)
    print(f"GPU available: {torch.cuda.is_available()}")

    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200,
        'imagenet': 1000,
    }
    s3_logger = init_s3_logger(config)

    # build model
    model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb = try_cuda(mb)
    mb.model.apply(weights_init)

    # ====================================== get dataloader ======================================
    data_dir = os.path.join(hydra.utils.get_original_cwd(), config.data_dir)
    data_dir = os.path.normpath(data_dir)
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, data_dir, config.subsample_ratio)

    # ========== get initial metrics =================
    s3_logger.add_table('pruning_metrics')
    s3_logger.log(dict(num_params=mb.num_params, num_ops=mb.num_ops), 'init', 'pruning_metrics')

    print("========== pruning ops ========================")
    mb = prune_ops(config, mb, trainloader, classes[config.dataset])
    s3_logger.log(dict(num_params=mb.num_params, num_ops=mb.num_ops), 'prune_ops', 'pruning_metrics')

    print("========== pruning weights ====================")
    mb = prune_weights(config, mb, trainloader, classes[config.dataset])
    prune_metrics = mb.get_ratio_at_each_layer()
    s3_logger.log(dict(num_params=prune_metrics['remaining_params'], num_ops=mb.num_ops),
                  'prune_weights', 'pruning_metrics')

    print(pd.DataFrame(s3_logger.data['pruning_metrics']).to_markdown())
    s3_logger.write_csv()

    # ========== finetuning =======================
    training_loop(
        net=mb.model,
        trainloader=trainloader,
        testloader=testloader,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        s3_logger=s3_logger
    )


if __name__ == '__main__':
    main()
