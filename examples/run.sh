#!/bin/bash

set -x
export PYTHONUNBUFFERED=1
export MKL_SERVICE_FORCE_INTEL=1
export RAY_storage_monitor_disk_usage_threshold=0.99

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

TRAIN_DATA_PATH=/examples/train_split.jsonl
VALIDATION_DATA_PATH=examples/val_subset_200.jsonl

export DEBUG_REWARD=0

python3 -m verl.trainer.main \
    config=examples/config_debug.yaml \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VALIDATION_DATA_PATH} \
    worker.actor.model.model_path=${MODEL_PATH}

