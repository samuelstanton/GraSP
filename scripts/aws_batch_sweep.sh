#!/bin/bash

cd ../BatchExperiments

script_name="image_classification"
batch_name="batch_sweep-op_pruner.type-op_pruner.target_percent-dataset"
datasets=("cifar10" "cifar100")
op_pruner_types=("grasp" "random" "weight_mag")
op_pruner_target_percents=(70 75 80 82 84 86 88 90 92 94)

num_trials=3

for dataset in ${datasets[@]}
do
  for prune_type in ${op_pruner_types[@]}
  do
    for op_percent in ${op_pruner_target_percents[@]}
    do
      job_name="${prune_type}-${op_percent}-${dataset}"
      exp_name="${batch_name}/${job_name}"
      echo "launching experiment: ${exp_name}"
      hydra_overrides="op_pruner.type=${prune_type},op_pruner.target_percent=${op_percent},dataset=${dataset},exp_name=${exp_name}"
      python launchers/grasp_launcher.py --script ${script_name} --hydra-overrides="${hydra_overrides}" \
      --num-jobs ${num_trials} --job-name ${job_name}
    done
  done
done