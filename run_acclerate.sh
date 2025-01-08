#!/usr/bin/env bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mamba2
which deepspeed
which python

# The new versions of CUDA and PyTorch require the following environment to be set up
# - NCCL_SOCKET_IFNAME. to find the proper value, using command line: `ip addr`
# - NCCL_IB_DISABLE. if the network-interface is not InfiniBand, set NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="lo"    # `lo` refers to loopback interface which applies to single machine
export NCCL_IB_DISABLE=1        # disable IB communication
export NCCL_P2P_DISABLE=1       # disable P2P communication

unset CUDA_VISIBLE_DEVICES
export PORT=$(python -Bu get_tcp_port.py 2>/dev/null | grep 'Distributed TCP PORT' | awk -F'|' '{print $2}' | xargs -n1 echo | head -n1)

# rm -rf results/multiome_BMMC_sortby_chrombatchsize64projection_dim64/checkpoints

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 \
#     --config_file config_files/accelerate_config.yaml \
#     train_accelerate.py \
#         --batch_size 64 \
#         --data_dir datasets/multiome/multiome_BMMC_sortby_chrom.h5mu \
#         --n_top_genes 20480 \
#         --n_top_peaks 40960 \
#         --binning 200 \
#         --config config_files/mamba2attn_config.json \
#         --epoch_nums 80 \
#         --results_dir results     

# python inference_accelerate.py \
#     --device cuda:1 \
#     --checkpoints results/multiome_BMMC_sortby_chrombatchsize64projection_dim64/checkpoints/scMamba.pt \
#     --batch_size 64 \
#     --data_dir datasets/multiome/multiome_BMMC_sortby_chrom.h5mu \
#     --n_top_genes 20480 \
#     --n_top_peaks 40960 \
#     --binning 200 \
#     --config config_files/mamba2attn_config.json \
#     --epoch_nums 80 \
#     --results_dir results  
rm -rf results/scale/fetal_10000batchsize64projection_dim32/checkpoints/

CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    --main_process_port $PORT \
    --config_file config_files/accelerate_config.yaml \
    train_accelerate.py \
        --batch_size 64 \
        --data_dir datasets/multiome/fetal_sample/fetal_10000.h5mu \
        --n_top_genes 10240 \
        --n_top_peaks 20480 \
        --binning 200 \
        --config config_files/mamba2_config.json \
        --epoch_nums 100 \
        --results_dir results/scale  

python inference_accelerate.py \
    --device cuda:1 \
    --checkpoints results/scale/fetal_10000batchsize64projection_dim32/checkpoints/scMamba.pt \
    --batch_size 64 \
    --data_dir datasets/multiome/fetal_sample/fetal_10000.h5mu \
    --n_top_genes 10240 \
    --n_top_peaks 20480 \
    --binning 200 \
    --config config_files/mamba2_config.json \
    --epoch_nums 100 \
    --results_dir results/scale