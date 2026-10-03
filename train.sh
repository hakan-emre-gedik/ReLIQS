#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on each participating machine.
# Usage: bash train.sh [config] [run_dir]
#
# Default: one machine with four GPUs.
#
# Example: four machines with one GPU each.
# Run on each machine, setting NODE_RANK to 0, 1, 2, or 3:
# NNODES=4 GPUS_PER_NODE=1 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#   bash train.sh options/main_training/koniq.json runs/koniq
#
# MASTER_ADDR must identify the rank-0 machine and be reachable
# from all machines. Use the same MASTER_PORT on every machine.
# Adjust per-GPU batch sizes when changing the total GPU count.

CONFIG="${1:-options/main_training/koniq.json}"
RUN_DIR="${2:-runs/koniq}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "$NNODES" == "1" ]]; then
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
    : "${NODE_RANK:?Set NODE_RANK for this machine (0 to NNODES-1)}"
    : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 machine address}"
fi

mkdir -p "$RUN_DIR"

torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    main.py \
    --config "$CONFIG" \
    --run_dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train_node_${NODE_RANK}.log"