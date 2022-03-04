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
from openselfsup.utils import (get_root_logger, dist_forward_collect_with_teacher,collect_env,nondist_forward_collect_with_teacher, traverse_replace)
# gaia lib
import gaiavision
import gaiassl

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('teacher_ckpt', help='checkpoint file')
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
    log_file = osp.join(cfg.work_dir, 'train_{}.log'.format(timestamp))
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
    logger.info(f'teacher_ckpt: {args.teacher_ckpt}')

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu = cfg.data.imgs_per_gpu,
        workers_per_gpu = cfg.data.workers_per_gpu,
        dist = distributed,
        shuffle=False
    )

    
    # build the model and load checkpoint
    model = build_model(cfg.model)
    teacher_model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    load_checkpoint(teacher_model, args.teacher_ckpt, map_location='cpu')
    #pdb.set_trace()
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        teacher_model = MMDataParallel(teacher_model, device_ids=[0])
        outputs = single_gpu_extract(model, teacher_model, data_loader, logger, args)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        teacher_model = MMDistributedDataParallel(
            teacher_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_extract(model, teacher_model, data_loader, logger, args)  # dict{key: np.ndarray}

    rank, _ = get_dist_info()
    if rank == 0:
        # 结果打印，并输出到日志中
        pass
            
def single_gpu_extract(model, teacher_model, data_loader, logger, args):
    #model.eval() # BN 确实是需要校正的。。。
    func = lambda **x: (model(mode='extract', **x),teacher_model(mode='extract',**x))
    results = nondist_forward_collect_with_teacher(func, data_loader,
                                      len(data_loader.dataset))
    

def multi_gpu_extract(model, teacher_model, data_loader, logger, args):
    #model.eval()
    func = lambda **x: model(mode='extract', **x)
    func_teacher = lambda **x: model(mode='extract', **x)
    rank, world_size = get_dist_info()
    #pdb.set_trace()
    results = dist_forward_collect_with_teacher(func, func_teacher, data_loader, rank,
                                   len(data_loader.dataset))
    #pdb.set_trace()
    
    if rank == 0:
        # ref:https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N       
        value = results['ressl_loss'].mean()
        logger.info(f'mean ressl loss: {value}')
    torch.distributed.barrier()


def similarity_compute(teacher_results,student_results,logger, args):
    pass

if __name__ == '__main__':
    main()
