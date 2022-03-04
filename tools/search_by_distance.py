# standard lib
import os
import pdb
import time
import json
import copy
import argparse
import importlib
import os.path as osp
from copy import deepcopy

# 3rd-parth lib
import torch
import torch.distributed as dist

# mm lib
import mmcv
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info, load_state_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from openselfsup import __version__
from openselfsup.apis import set_random_seed
from openselfsup.datasets import build_dataset, build_dataloader
from openselfsup.models import build_model
from openselfsup.utils import collect_env, get_root_logger, traverse_replace

# gaia lib
import gaiavision
from gaiavision import broadcast_object
from gaiavision.model_space import (ModelSpaceManager,
                                    build_sample_rule,
                                    build_model_sampler,
                                    unfold_dict,
                                    fold_dict)

import gaiassl
from gaiassl.datasets import ScaleManipulator, manipulate_dataset
from gaiassl.apis import multi_gpu_test_with_distance


DISTANCES = {
    'mse': torch.nn.MSELoss,
    'kl': torch.nn.KLDivLoss,
    'ressl':torch.nn.KLDivLoss # 这个loss到底咋用还是没太确定，主要是有个log操作
}


def parse_args():
    parser = argparse.ArgumentParser(description='Search a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='train config file path')
    parser.add_argument(
        '--from_ssl',
        type=bool,
        default=True,
        help='whether pretrained weights are imported from ssl')
    parser.add_argument(
        '--model_space_path',
        type=str,
        help='path of file that records model information')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--out-name', default='metrics.json', help='output result file name')
    parser.add_argument('--metric-tag', default='distance',
                        help='tag of metric in search process')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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

    metric_dir = os.path.join(args.work_dir, 'search_subnet')
    args.metric_dir = metric_dir

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.gpus = args.gpus

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        assert cfg.model.type not in \
            ['DeepCluster', 'MOCO', 'SimCLR', 'ODC', 'NPID'], \
            "{} does not support non-dist training.".format(cfg.model.type)
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    os.makedirs(args.metric_dir, exist_ok=True)
    save_path = os.path.join(args.metric_dir, args.out_name)
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

    # load model information, CLI > cfg
    if args.model_space_path is not None:
        cfg.model_space_path = args.model_space_path
    assert cfg.get('model_space_path', None) is not None
    logger.info('Model space:\n{}'.format(cfg.model_space_path))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # prepare model and pretrained weights
    model = build_model(cfg.model)
    ckpt = torch.load(args.checkpoint)['state_dict']
    '''
    # 没太get到这个地方是要做什么
    if args.from_ssl:
        out_ckpt = dict()
        for key, value in ckpt.items():
            if key.startswith('backbone'):
                out_ckpt[key] = value
            elif key.startswith('encoder_q.1'):
                out_ckpt['neck' + key[11:]] = value
        ckpt = out_ckpt
    '''
    load_state_dict(model, ckpt)
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    # collect model of interests
    sampled_model_metas = []
    model_space = ModelSpaceManager.load(cfg.model_space_path)
    rule = build_sample_rule(cfg.model_sampling_rules)
    sub_model_space = model_space.ms_manager.apply_rule(rule)
    model_metas = sub_model_space.ms_manager.pack()
    if rank == 0:
        print("Please notice, the encoder_k always keeps largest architecture")
    for each in model_metas:
        each['arch'].pop('encoder_k')
    #pdb.set_trace()
    # set up distance
    cfg_distance = cfg.get('distance', {'type': 'kl', 'kwargs': {}})
    distance = DISTANCES[cfg_distance['type']](**cfg_distance['kwargs'])
    
    '''
    当时flops.json提取有问题，有重复值。。。删除一下
    '''
    model_metas_no_id = copy.deepcopy(model_metas)
    for each in model_metas_no_id:
        each.pop('index')
    new_model_metas_no_id = []
    new_model_metas = []
    
    for each_no_id, each in zip(model_metas_no_id, model_metas):
        
        if each_no_id not in new_model_metas_no_id:
            new_model_metas.append(each)
            new_model_metas_no_id.append(each_no_id)
    model_metas = new_model_metas
    

    #pdb.set_trace()
    for i, model_meta in enumerate(model_metas):
        # sync model_meta between ranks
        model_meta = broadcast_object(model_meta)
        
        # manipulate data
        # 暂时先不manipulate data
        #input_shape = model_meta['data']['input_shape']
        #new_scale = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape
        #scale_manipulator = ScaleManipulator(new_scale)
        #manipulate_dataset(cfg.data.val, scale_manipulator)

        dataset = build_dataset(cfg.data.train)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True,
            shuffle=False)

        #pdb.set_trace()

        # TODO:run test
        print("start running")
        outputs = multi_gpu_test_with_distance(model, model_meta, data_loader, distance, rank)
        dist.barrier()
        #pdb.set_trace()
        result_model_meta = deepcopy(model_meta)
        metrics = {}
        
        if rank == 0:
            # TODO: replace the ugly workaround
            koi = ['mean']
            for name in koi:
                metrics[name] = outputs[name]

            metric_meta = result_model_meta.setdefault('metric', {})
            metric_meta[args.metric_tag] = metrics
            result_model_meta['metric'] = metric_meta
            sampled_model_metas.append(result_model_meta)
            logger.info('-- model meta:')
            logger.info(json.dumps(sampled_model_metas[-1], indent=4))
        dist.barrier()

    if rank == 0:
        sub_model_space = ModelSpaceManager.load(sampled_model_metas)
        sub_model_space.ms_manager.dump(save_path)


if __name__ == '__main__':
    main()
