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

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp --test_pretrained \
    -a resnet56 --dataset cifar10 \
    --pretrained_ckpt ./Experiments/scratch__resnet56__cifar10_SERVER230-20240719-093254/weights/ckpt_best.pth \
    --pruner oracle --prune_method oracle --prune_with_hooks \
    --batch_size 256 \
    --stage_pr *layer*conv1*:0.5 \
    --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
    --weight_decay 0.0005 \
    --batch_oracle 0 \
    --random yes \
    --save_combination yes \
    --experiment_name oracle__resnet56__cifar10__conv10.5__lr2__savecombination

# Repeat the same experiment for 3 times
for t in {1..3}
do
    # Multi-GPU
    for i in {0..7}
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
            -a resnet56 --dataset cifar10 \
            --pretrained_ckpt ./Experiments/scratch__resnet56__cifar10_SERVER230-20240719-093254/weights/ckpt_best.pth \
            --pruner oracle --prune_method oracle --prune_with_hooks \
            --batch_size 256 \
            --stage_pr *layer*conv1*:0.5 \
            --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
            --weight_decay 0.0005 \
            --batch_oracle $i \
            --random yes \
            --save_combination no \
            --experiment_name oracle__resnet56__cifar10__conv10.5__lr2__t$t\_batch$i > ./record/resnet56_conv10.5_lr2_t$t\_batch$i.txt 2>&1 &
    done
    wait

    CUDA_VISIBLE_DEVICES=6 python main.py --no_ddp --no_scp --test_pretrained \
        -a resnet56 --dataset cifar10 \
        --pretrained_ckpt ./Experiments/scratch__resnet56__cifar10_SERVER230-20240719-093254/weights/ckpt_best.pth \
        --pruner l1 --prune_method l1 --prune_with_hooks \
        --batch_size 256 \
        --stage_pr *layer*conv1*:0.5 \
        --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
        --weight_decay 0.0005 \
        --experiment_name l1__resnet56__cifar10__conv10.5__lr2__t$t > ./record/resnet56_conv10.5_lr2_l1_t$t.txt &    
done
