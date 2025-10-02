#!/bin/bash

set -x
export PYTHONUNBUFFERED=1
# export RAY_TMPDIR=/data3/xiongyuqi/ray_temp
# mkdir -p /data3/xiongyuqi/ray_temp
export MKL_SERVICE_FORCE_INTEL=1
export RAY_storage_monitor_disk_usage_threshold=0.99
# export NCCL_CUMEM_ENABLE=1
# --- 模型和数据路径 (保持不变) ---
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

TRAIN_DATA_PATH=/examples/train_split.jsonl
VALIDATION_DATA_PATH=examples/val_subset_200.jsonl


# --- 创建调试用的数据集 (保持不变) ---
# head -n 10 ${TRAIN_DATA_PATH} > ${DEBUG_TRAIN_PATH} # 用更小的数据集 (10条)
# head -n 10 ${VALIDATION_DATA_PATH} > ${DEBUG_VAL_PATH}

# =========================================================================
#  ★ 启动训练 (调试模式) ★
# =========================================================================
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# --- 调试策略 1: 运行 Dummy 模式 ---
# 这个模式不调用外部API，如果这个能跑通且奖励不为0，说明框架和数据流没问题。
echo "--- LAUNCHING IN DUMMY MODE ---"
# export DEBUG_REWARD=1
# export NEBIUS_API_KEY="YOUR_REAL_API_KEY_HERE" # Dummy 模式不需要 Key

# python3 -m verl.trainer.main \
#     config=examples/config_debug.yaml \
#     data.train_files=${DEBUG_TRAIN_PATH} \
#     data.val_files=${DEBUG_VAL_PATH} \
#     worker.actor.model.model_path=${MODEL_PATH}
    # 其他参数已在 config_debug.yaml 中设置

# --- 调试策略 2: 运行真实模式 (如果 Dummy 模式成功后再尝试) ---
# 注释掉上面的 Dummy 模式命令块，然后取消下面的注释来运行真实模式。
# echo "--- LAUNCHING IN REAL MODE ---"
export DEBUG_REWARD=0
# export NEBIUS_API_KEY="YOUR_REAL_API_KEY_HERE" # 真实模式需要 Key
#
python3 -m verl.trainer.main \
    config=examples/config_debug.yaml \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VALIDATION_DATA_PATH} \
    worker.actor.model.model_path=${MODEL_PATH}


