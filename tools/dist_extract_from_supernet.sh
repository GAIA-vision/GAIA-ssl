#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

SRC_CKPT=$1
OUT_CKPT=$2
CFG=$3
GPUS=$4
PY_ARGS=${@:5}
PORT=${PORT:-29500}

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# --work_dir $WORK_DIR --seed 0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/extract_from_supernet.py $SRC_CKPT $OUT_CKPT $CFG \
    --launcher pytorch ${PY_ARGS}
