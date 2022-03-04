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
        description='MMDet test (and eval) a model')
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
    parser.add_argument('--t', type=float, default=0.1, help='temperature for knn predict')
    parser.add_argument('--k', type=int, default=200, help='k nearest neighbors for knn predict')
    parser.add_argument('--t_sne', type=bool, default=False, help='Whther t-sne visualization')
    parser.add_argument('--visualize_category', type=int, default=10, help='The max classes num to visualize')
    parser.add_argument('--classes', type=int, default=1000, help='Dataset classes num(defaul 1k for ImageNet)')
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
    save_path = os.path.join(cfg.work_dir,'t-sne.png')
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_knn(model, train_data_loader, val_data_loader, logger, args)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_knn(model, train_data_loader, val_data_loader,logger, args)  # dict{key: np.ndarray}

    rank, _ = get_dist_info()
    if rank == 0:
        # 结果打印，并输出到日志中
        pass
            
def single_gpu_knn(model, train_data_loader, val_data_loader, logger, args):
    model.eval() # BN 确实是需要校正的。。。
    func = lambda **x: model(mode='get_embedding', **x)
    train_results = nondist_forward_collect(func, train_data_loader,
                                      len(train_data_loader.dataset))
    val_results = nondist_forward_collect(func, val_data_loader,
                                      len(val_data_loader.dataset))
    knn_predict(train_results, val_results, logger, args)

def multi_gpu_knn(model, train_data_loader, val_data_loader, logger, args):
    model.eval()
    func = lambda **x: model(mode='get_embedding', **x)
    rank, world_size = get_dist_info()
    train_results = dist_forward_collect(func, train_data_loader, rank,
                                   len(train_data_loader.dataset))
    val_results = dist_forward_collect(func, val_data_loader, rank,
                                   len(val_data_loader.dataset))
    if rank == 0:
        # ref:https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N       
        knn_predict(train_results, val_results, logger, args)
    torch.distributed.barrier()
def t_sne_visualize(embedding, label, save_path='./t-sne.png', visualize_category=10, n_components=2,init='pca',random_state=614,**kwargs):
    '''
    embedding: [N,D] torch.tensor
    label: [N,] ndarray

    tenor   -> ndarray  tensor.numpy()
    ndarray -> tensor   torch.tensor(ndarray)
    '''
    random.seed(36)
    pdb.set_trace()
    unique_label = list(set(label))
    
    classes_num = len(unique_label)
    choices = set()
    while(len(choices)<visualize_category):
        choices.add(unique_label[random.randint(0, classes_num-1)])
    visualize_embedding = []
    visualize_label = []
    for each_choice in choices:
        visualize_embedding.append(embedding[label==each_choice])
        visualize_label.append(label[label==each_choice])
    pdb.set_trace()
    embedding = torch.cat(visualize_embedding,dim=0)
    label = np.concatenate(visualize_label, axis=0)
    label = label.reshape(-1).astype(np.int64)
    temp_label = copy.deepcopy(label)
    label_map = {}
    for idx,each in enumerate(set(temp_label)):
        label_map[each] = idx

    for old_label,new_label in label_map.items():
        label[temp_label==old_label] = new_label
    pdb.set_trace()
    embedding = F.normalize(embedding, dim=1).numpy()
    tsne = sklearn.manifold.TSNE(n_components=n_components, init=init, random_state=random_state, **kwargs)
    embedding = tsne.fit_transform(embedding) # [N,D] -> [N,n_components]
    x_min, x_max = embedding.min(0), embedding.max(0)
    embedding_norm = (embedding-x_min)/(x_max-x_min)
    train_len = embedding_norm.shape[0]
    for i in range(train_len):
        plt.text(embedding_norm[i,0],embedding_norm[i,1],str(label[i]),color=plt.cm.Set1(label[i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)

def feature_statistic(embedding, label, classes, logger):
    '''
    train_embedding: [N,D] tensor
    tarin_label: [N,] tensor,dtype=torch.long()
    '''

    # 把同一label的feature算一个mean，得到各个类别的ceneter feature [C,D]
    per_center_feature = torch.zeros(classes,embedding.shape[1],dtype=embedding.dtype)
    label_count = torch.zeros(classes,1,dtype=embedding.dtype)
    for idx,each_label in enumerate(label):
        per_center_feature[each_label] += embedding[idx]
        label_count[each_label] += 1
    per_center_feature = per_center_feature / label_count
    # 算一个矩阵，各个centerfeature之间的距离 [C,C]
    per_center_distance = torch.mm(per_center_feature, per_center_feature.T)
    diag = torch.diag(per_center_distance)
    per_center_distance = per_center_distance-torch.diag_embed(diag)
    # logger记录下每个center 之间的distance mean还有就是minimum distance 和 maximum distance
    logger.info('All distance is calculated by cosine similarity. Large means more near')
    logger.info(f'center_distance_mean: {torch.sum(per_center_distance)/classes/classes}')
    for i in range(classes):
        logger.info(f'center_{i} over other center distance: average({torch.sum(per_center_distance[i])/classes}),max({torch.max(per_center_distance[i])}),min({torch.min(per_center_distance[i])})')
    
    # 求一下每个类别的所有feature 到center feature距离的平均值 [C] 用torch.gather好像更好一些。
    center_feature_copy = torch.zeros(embedding.shape,dtype=embedding.dtype)
    train_len = embedding.shape[0]
    for i in range(train_len):
        center_feature_copy[i] = per_center_feature[label[i]]
    out_center_distance = torch.sum(center_feature_copy * embedding,1)

    mean_out_center_distance = torch.tensor(0.)
    for i in range(classes):
        temp_out_center_distance = torch.sum(out_center_distance[label==i])/torch.sum(label==i)
        mean_out_center_distance += temp_out_center_distance
        logger.info(f"Classes {i}: mean out of center distance: {temp_out_center_distance}")
    logger.info(f'Mean out of center distance: {mean_out_center_distance/classes}')

def knn_predict(train_results, val_results, logger, args):
    knn_t = args.t
    knn_k = args.k
    classes = args.classes
    t_sne = args.t_sne
    save_path=args.save_path
    visualize_category = args.visualize_category

    logger.info(f'knn.T: {knn_t}')
    logger.info(f'knn.k: {knn_k}\n')
    if t_sne:  
        if classes == 1000:   
            print("Warning: ImageNet Train cost unaffordable time cost")
        train_embedding = torch.tensor(train_results['variable_0'])
        train_label = train_results['variable_1'].astype(np.int64)
        t_sne_visualize(train_embedding, train_label, save_path=save_path, visualize_category=10)

    train_embedding = torch.tensor(train_results['variable_0']) # [N, D]
    train_label = torch.tensor(train_results['variable_1']).view(-1).long() # [N,]
    val_embedding = torch.tensor(val_results['variable_0']) # [N1, D]
    val_label = torch.tensor(val_results['variable_1']).view(-1).long()  # [N1,]
    # 这个normalize很重要，MoCo是有明确的normalize，为什么BYOL没有但依旧训练稳定？
    train_embedding = F.normalize(train_embedding, dim=1)
    val_embedding = F.normalize(val_embedding, dim=1)

    ##################### Feature Center Statistic #####################
    feature_statistic(train_embedding, train_label, classes, logger)
    #####################                          #####################

    #[N1, N]
    sim_matrix = torch.mm(val_embedding, train_embedding.T)
    # [N1, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [N1, K]             val_label.expand(train_embedding.size(0), -1) [N] -> [N1, N] 每一列元素相同
    sim_labels = torch.gather(train_label.expand(val_embedding.size(0), -1), dim=-1, index=sim_indices)
    # sim_weight = e^(weight/t)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class [N1*k, C]
    one_hot_label = torch.zeros(val_embedding.size(0) * knn_k, classes, device=sim_labels.device)
    # [N1*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [N, C]                  torch.sum([N1,K,C]*[N1,K,1],dim=1)  每个类别按照对应的相似度获得投票，加权投票，不是平均意义
    pred_scores = torch.sum(one_hot_label.view(val_embedding.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    print("Get acc: ", torch.sum(pred_labels[:,0]==val_label)/len(val_label)*100, end='%\n')    
    logger.info(f'Get acc: {torch.sum(pred_labels[:,0]==val_label)/len(val_label)*100}%')

if __name__ == '__main__':
    main()
