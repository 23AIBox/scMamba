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


CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes=2 \
    --config_file config_files/accelerate_config.yaml --main_process_port $PORT \
    train_accelerate.py \
        --batch_size 64 \
        --data_dir datasets/multiome/PBMC.h5mu  \
        --config config_files/scmamba2_config.json \
        --n_top_genes 10240 \
        --n_top_peaks 20480 \
        --epoch_nums 80 \
        --results_dir results/benckmark

python inference.py \
    --device cuda:5 \
    --checkpoints results/benckmark/PBMCbatchsize64emb_dim64/checkpoints/scMamba.pt \
    --batch_size 64 \
    --data_dir datasets/multiome/PBMC.h5mu  \
    --config config_files/scmamba2_config.json \
    --n_top_genes 10240 \
    --n_top_peaks 20480 \
    --epoch_nums 80 \
    --results_dir results/benckmark

