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


CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes=2 \
    --config_file config_files/accelerate_config.yaml --main_process_port $PORT \
    train_accelerate_rna_adt.py \
        --batch_size 64 \
        --data_dir datasets/multiome/citeseq_BMMC_S1.h5mu \
        --batch_key batch \
        --n_top_genes 10240 \
        --binning 51 \
        --config config_files/scmamba2_config_rna_adt.json \
        --epoch_nums 80 \
        --results_dir results/benckmark

python inference_rna_adt.py \
    --checkpoints results/benckmark/citeseq_BMMC_S1batchsize64emb_dim64/checkpoints/scMamba.pt \
    --device cuda:4 \
    --batch_size 64 \
    --data_dir datasets/multiome/citeseq_BMMC_S1.h5mu \
    --batch_key batch \
    --n_top_genes 10240 \
    --binning 51 \
    --config config_files/scmamba2_config_rna_adt.json \
    --epoch_nums 80 \
    --results_dir results/benckmark 