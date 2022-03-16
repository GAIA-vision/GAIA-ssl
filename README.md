# GAIA-ssl
An AutoML toolbox specialized in contrastive learning. 
# Install

  ## requirements:
  torch 1.8.0
  
  gaiavision
  
  mmcv-full 1.3.0

# Command
  ## Supernet training
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh apps/dynmoco/configs/local/ar50to101_10pc_bs64_200_epoch.py 8
  ```

  ## Feature similarity computatoin
  For classification downstream tasks:
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_search.sh apps/dynmoco/configs/local/supernet_search.py /path/to_ckpt workdir 4
  ```
  For dense prediction downstream tasks:
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_search.sh apps/dynmoco/configs/local/supernet_dense_search.py /path/to_ckpt workdir 4
  ```

  ## Extract subnet
  ``

# Citation

If you find this project useful in your research, please consider cite:

