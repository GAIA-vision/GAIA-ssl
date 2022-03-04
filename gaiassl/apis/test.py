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


def multi_gpu_test_with_distance(model, meta, data_loader, distance, rank):
    
    results = torch.tensor(0.).cuda()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))
    model.module.manipulate_arch(meta['arch'])
    query_embeddings = None
    key_embeddings = None
    temperature = 1
    #pdb.set_trace()
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            # manipulate arch
            result1 = model(**data, mode='get_embedding')  
            result2 = model(**data, mode='get_embedding', extract_from='encoder_k')
            batch_size = result1.size(0)
            if query_embeddings is None:
                query_embeddings = torch.zeros(result1.size(1),len(data_loader)*batch_size).cuda() # [C, L] 
                key_embeddings = torch.zeros(result2.size(1),len(data_loader)*batch_size).cuda() # [C, L]
            query_embeddings[:,idx*batch_size:(idx+1)*batch_size] = result1.T
            key_embeddings[:,idx*batch_size:(idx+1)*batch_size] = result2.T

            logits_q = torch.einsum('nc,ck->nk', [result1, key_embeddings])
            logits_k = torch.einsum('nc,ck->nk', [result2, key_embeddings])
            results += - torch.sum( F.softmax(logits_k / temperature, dim=1) \
                                             * F.log_softmax(logits_q / temperature, dim=1), dim=1).mean()
        if rank == 0:
            sys.stdout.flush()
            prog_bar.update()
    results = results/len(data_loader)
    results_all = {}
    dist.barrier()
    dist.all_reduce(results)
    world_size = dist.get_world_size()
    results_all['mean'] = (results / world_size).item()
    return results_all
