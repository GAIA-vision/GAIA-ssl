#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# --work_dir $WORK_DIR --seed 0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --launcher pytorch ${PY_ARGS}
