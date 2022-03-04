#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
GPUS=$3
PORT=${PORT:-29500}

if [ ! -d $WORK_DIR ];then
    mkdir $WORK_DIR
fi
cp $CONFIG $WORK_DIR

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/count_flops.py \
    ${CONFIG} \
    --work-dir=$WORK_DIR \
    --launcher pytorch 

