# standard lib
import sys
import pdb

# 3rd-party lib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# mm lib
import mmcv

DISTANCES = ['mse','cosine','relation']

def multi_gpu_test_with_distance(model, meta, teacher_data_loader, student_data_loader, distance_metric, rank):
    assert distance_metric in DISTANCES
    results = torch.tensor(0.).cuda()
    results_qq = torch.tensor(0.).cuda()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(teacher_data_loader))
    model.module.manipulate_arch(meta['arch'])
    query_embeddings = None
    key_embeddings = None
    temperature = 0.2

    for idx, data in enumerate(zip(teacher_data_loader, student_data_loader)):

        with torch.no_grad():
            # scale up for avoiding overflow
            result1 = model(**data[0], mode='get_embedding')*100 
            result2 = model(**data[0], mode='get_embedding', extract_from='encoder_k')*100

            result1 = torch.nn.functional.normalize(result1, dim=1)
            result2 = torch.nn.functional.normalize(result2, dim=1)
            batch_size = result1.size(0)
            if distance_metric == 'mse':
                results += torch.sum((result1-result2)*(result1-result2))/batch_size
            if distance_metric == 'cosine':
                results += torch.sum(torch.einsum('nc,nc->n', [result1,result2]))/batch_size
            elif distance_metric == 'relation':
                if query_embeddings is None:
                    query_embeddings = torch.zeros(result1.size(1),len(teacher_data_loader)*batch_size).cuda() # [C, L] 
                    key_embeddings = torch.zeros(result2.size(1),len(teacher_data_loader)*batch_size).cuda() # [C, L]
                query_embeddings[:,idx*batch_size:(idx+1)*batch_size] = result1.T
                key_embeddings[:,idx*batch_size:(idx+1)*batch_size] = result2.T
                
                logits_q = torch.einsum('nc,ck->nk', [result1, key_embeddings[:,:(idx+1)*batch_size]])
                logits_q_qq = torch.einsum('nc,ck->nk', [result1, query_embeddings[:,:(idx+1)*batch_size]])
                logits_k = torch.einsum('nc,ck->nk', [result2, key_embeddings[:,:(idx+1)*batch_size]])
                results += - torch.sum( F.softmax(logits_k / temperature, dim=1) \
                                                * F.log_softmax(logits_q / temperature, dim=1), dim=1).mean()
                results_qq += - torch.sum( F.softmax(logits_k / temperature, dim=1) \
                                                * F.log_softmax(logits_q_qq / temperature, dim=1), dim=1).mean()
        if rank == 0:
            sys.stdout.flush()
            prog_bar.update()
    results = results/len(teacher_data_loader)
    if distance_metric == 'relation':
        results_qq = results_qq/len(teacher_data_loader)

    results_all = {}
    dist.barrier()
    dist.all_reduce(results)
    dist.all_reduce(results_qq)
    world_size = dist.get_world_size()

    if distance_metric == 'mse':
        results_all['mse'] = (results / world_size).item()
    elif distance_metric == 'cosine':
        results_all['cosine'] = (results / world_size).item()
    elif distance_metric == 'relation':
        results_all['qk_kk_mean'] = (results / world_size).item()
        results_all['qq_kk_mean'] = (results_qq / world_size).item()
    return results_all


def multi_gpu_test_with_dense_distance(model, meta, teacher_data_loader, student_data_loader, distance_metric, rank):
    assert distance_metric in DISTANCES
    results = torch.tensor(0.).cuda()
    results_qq = torch.tensor(0.).cuda()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(teacher_data_loader))
    model.module.manipulate_arch(meta['arch'])
    query_embeddings = None
    key_embeddings = None
    temperature = 0.2
    #pdb.set_trace()
    for idx, data in enumerate(zip(teacher_data_loader, student_data_loader)):
        #pdb.set_trace()
        with torch.no_grad():

            result1 = model(**data[0], mode='extract') #([N,C,H,W],...) depend on your config, which stage's feature will be return. 
            result2 = model(**data[0], mode='extract', extract_from='encoder_k') #([N,C,H,W],...)

            for tensor_1, tensor_2 in zip(result1, result2): 
                # scale up for avoiding overflow
                # resolution of tensor from early stage may cause memory out. Consider cropping it
                tensor_1 = tensor_1*100
                tensor_2 = tensor_2*100
                
                tensor_1 = torch.nn.functional.normalize(tensor_1, dim=1)
                tensor_2 = torch.nn.functional.normalize(tensor_2, dim=1)

                tensor_1 = tensor_1.view(tensor_1.size(0),tensor_1.size(1),-1) # [N,C,H*W]
                tensor_2 = tensor_2.view(tensor_2.size(0),tensor_2.size(1),-1) # 
                
                tensor_1 = torch.bmm(tensor_1.transpose(1,2),tensor_1) #[N,H*W,H*W]
                tensor_1 = tensor_1.view(-1, tensor_1.size(2)) # [N*HW, H*W]
                tensor_2 = torch.bmm(tensor_2.transpose(1,2),tensor_2) 
                tensor_2 = tensor_2.view(-1, tensor_2.size(2)) 

                batch_size = tensor_1.size(0)
                if distance_metric == 'mse':
                    results += torch.sum((tensor_1-tensor_2)*(tensor_1-tensor_2))/batch_size
                elif distance_metric == 'cosine':
                    results += torch.sum(torch.einsum('nc,nc->n', [tensor_1,tensor_2]))/batch_size
                elif distance_metric == 'kl':
                    results += torch.sum(- torch.sum( F.softmax(tensor_2 / temperature, dim=1) * F.log_softmax(tensor_1 / temperature, dim=1), dim=1))/batch_size
        if rank == 0:
            sys.stdout.flush()
            prog_bar.update()
    results = results/len(teacher_data_loader)

    results_all = {}
    dist.barrier()
    dist.all_reduce(results)
    world_size = dist.get_world_size()
    #pdb.set_trace()
    if distance_metric == 'mse':
        results_all['mse'] = (results / world_size).item()
    elif distance_metric == 'cosine':
        results_all['cosine'] = (results / world_size).item()
    elif distance_metric == 'kl':
        results_all['kl'] = (results / world_size).item()
    return results_all
