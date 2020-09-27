# Neural Architecture Search by Preserving Gradient Flow
This repo started as a fork of [Picking Winning Tickets Before Training by Preserving Gradient Flow](https://openreview.net/forum?id=SkgsACVKPH).

1. The code has been refactored to use [Hydra](https://hydra.cc/) for configuration. See the `hydra` directory for
all default behavior. Any Hydra config field can be overridden from the command-line. 

# Requirements
python 3.6, [graphviz](https://graphviz.org/download/)
```
conda create --name grasp-nas-env python=3.6
conda activate grasp-nas-env
sudo apt update
sudo apt install graphviz
pip install requirements.txt
pip install -e .
```

# Logging
By default, your hydra config and all outputs, including dataframes and checkpoints are logged locally in
`config.log_dir`. If you have AWS CLI configured, you can alternatively use `logger=s3` to save your results to 
`s3://${logger.params.bucket_name}/{logger.params.log_dir}`. 


# Datasets
1. By default, datasets are expected to be in `./data` (e.g. `./data/CIFAR10`, `./data/CIFAR100`).
You can override the default dataset directory using `dataset.dataset_dir`.
2. CIFAR-10 & CIFAR-100 will automatically be downloaded if not present.
3. Download tiny imagenet from "https://tiny-imagenet.herokuapp.com", and place it in ../data/TinyImageNet.
   Please make sure there will be two folders, `train` and `val`,  under the directory of `./data/TinyImageNet`.
   In either `train` or `val`, there will be 200 folders storing the images of each category.  <b>Or</b> You can also download the processed data from [here]( https://drive.google.com/file/d/1juoN5cRVa8I1TsfFMtsCec2Wy_0z646K/view?usp=sharing ).
4. MNIST is not currently supported, since it has single channel images (TODO)
5. ImageNet will no longer automatically download from PyTorch, so it must also be downloaded (TODO add links)

# Networks
Currently supported networks 
1. VGG (TODO add ref)
2. ResNet (TODO add ref)
3. GraphNet

# How to run?
```
# CIFAR-10, GraphNet32.75.50, (75% operation pruning ratio, 50% weight pruning ratio)
$ python scripts/image_classification.py dataset=cifar10 network=graphnet op_pruner.target_ratio=0.75 
weight_pruner.target_ratio=0.5

# CIFAR-100, ResNet32.90 (90% weight pruning ratio)
$ python scripts/image_classification.py dataset=cifar100 network=resnet weight_pruner.target_ratio=0.90
```
For the default behavior of all experiments, please refer to the `hydra` directory. Use command-line overrides 
`debug=True`, `subsample_ratio=0.125`, and `num_train_epochs=2` for quick testing.


# Citation
To cite this work, please use
```
@inproceedings{
Wang2020Picking,
title={Picking Winning Tickets Before Training by Preserving Gradient Flow},
author={Chaoqi Wang and Guodong Zhang and Roger Grosse},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SkgsACVKPH}
}
```

