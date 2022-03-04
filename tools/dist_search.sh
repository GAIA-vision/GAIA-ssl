#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
CKPT_PATH=$2
WORK_DIR=$3
GPUS=$4
PY_ARGS=${@:5}
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/search_by_distance.py ${CFG} ${CKPT_PATH} \
    --launcher pytorch ${PY_ARGS} \
    --work_dir ${WORK_DIR}

cp $CFG $WORK_DIR  

