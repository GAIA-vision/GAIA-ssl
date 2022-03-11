import pdb
import copy
import random
import argparse
import importlib
import os
import os.path as osp
import time
import copy

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.utils import (get_root_logger, dist_forward_collect,collect_env,
                               nondist_forward_collect, traverse_replace)
# gaia lib
import gaiavision
import gaiassl

def parse_args():
    parser = argparse.ArgumentParser(
        description='Openself test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument('--save_path', type=str, default='./t_sne.png', help='The figure save path of t-sne visualization')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'train_{} t-{} k-{}.log'.format(timestamp,args.t,args.k))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))
    logger.info(f'checkpoint: {args.checkpoint}')


    # build the train dataloader
    assert cfg.data_source_cfg.get('return_label',False), "Must return label in your train dataset config"
    train_dataset = build_dataset(cfg.data.train)
    train_data_loader = build_dataloader(
        train_dataset,
        imgs_per_gpu = cfg.data.imgs_per_gpu,
        workers_per_gpu = cfg.data.workers_per_gpu,
        dist = distributed,
        shuffle=False
    )
    val_dataset = build_dataset(cfg.data.val)
    val_data_loader = build_dataloader(
        val_dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    save_path = os.path.join(cfg.work_dir,'visual_lize.npz')
    model.eval()

if __name__ == '__main__':
    main()
