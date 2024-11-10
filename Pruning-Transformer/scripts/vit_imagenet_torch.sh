#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Function to kill all background jobs started by this script
terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

# Set up trap to call terminate() function on script exit or interruption
trap terminate SIGINT SIGTERM

# set visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

exp_prefix="vit-imagenet-tv"
seed_list=(2024)

for seed in ${seed_list[@]}; do
    echo "Start training with seed ${seed}"

    exp_name="${exp_prefix}-seed${seed}"

    # mkdir record folder
    mkdir -p ./record/${exp_name}

    accelerate launch --config_file ./default_config.yaml main_tv.py -a vit_b_16 --dataset imagenet \
        --data_path /datasets/fengsicheng \
        --experiment_name ${exp_name} \
        --batch_size 128 \
        --distributed \
        --bottleneck \
        --retrain_epoch 300 \
        --retrain_lr 0.00015 \
        --pruning_ratio 0.5 \
        --pruning_type oracle \
        --weight_decay 0.3 \
        --optimizer adam \
        --oracle_seed ${seed} \
        --lr_decay cosine \
        --label_smoothing 0.11 \
        --mixup_alpha 0.2 \
        --cutmix_alpha 1.0 &
    wait
done
