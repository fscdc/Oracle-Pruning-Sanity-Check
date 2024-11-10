#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

#!/bin/bash

# Function to kill all background jobs started by this script
terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

# Set up trap to call terminate() function on script exit or interruption
trap terminate SIGINT SIGTERM

if [ ! -d "./record" ]; then
    mkdir -p ./record
fi

# Repeat the same experiment for 3 times
for t in 3
do
    # Multi-GPU
    for i in {0..7}
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
            -a resnet18 --dataset imagenet \
            --data_path /root/autodl-tmp/ \
            --pruner oracle --prune_method oracle --prune_with_hooks \
            --batch_size 256 \
            --stage_pr *layer*conv1*:0.5 \
            --lr_ft 0:0.01,1:0.001,2:0.0001 --epochs 3 \
            --weight_decay 0.0001 \
            --batch_oracle $i \
            --random yes \
            --save_combination no \
            --num_missions 160 \
            --num_batch 8 \
            --experiment_name oracle__resnet18__imagenet__conv10.5__lr2__t$t\_batch$i > ./record/resnet18_conv10.5_lr2_t$t\_batch$i.txt &
    done
    wait 
done