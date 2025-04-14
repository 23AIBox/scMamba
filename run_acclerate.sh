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

# rm -rf results/benckmark/PBMCbatchsize128projection_dim16/checkpoints

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes=2 \
    --config_file config_files/accelerate_config.yaml --main_process_port $PORT \
    train_accelerate.py \
        --batch_size 64 \
        --data_dir datasets/multiome/PBMC.h5mu  \
        --config config_files/scmamba2_config.json \
        --epoch_nums 80 \
        --results_dir results/benckmark

python inference_accelerate.py \
    --device cuda:7 \
    --checkpoints results/benckmark/PBMCbatchsize64emb_dim64/checkpoints/scMamba.pt \
    --batch_size 64 \
    --data_dir datasets/multiome/PBMC.h5mu  \
    --config config_files/scmamba2_config.json \
    --epoch_nums 80 \
    --results_dir results/benckmark

