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

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp --test_pretrained \
    -a resnet50 --dataset imagenet \
    --data_path /datasets/fengsicheng/ \
    --pruner oracle --prune_method oracle --prune_with_hooks \
    --batch_size 256 \
    --stage_pr *conv1:0.4375,*layer1.0.conv1*:0.4375,*layer1.0.conv2*:0.4375,*layer1.1.conv1*:0.4375,*layer1.1.conv2*:0.4375,*layer1.2.conv1*:0.4375,*layer1.2.conv2*:0.4375,*layer2.0.conv1*:0.4375,*layer2.0.conv2*:0.4375,*layer2.1.conv1*:0.4375,*layer2.1.conv2*:0.4375,*layer2.2.conv1*:0.4375,*layer2.2.conv2*:0.4375,*layer2.3.conv1*:0.4375,*layer2.3.conv2*:0.4375,*layer3.0.conv1*:0.433,*layer3.0.conv2*:0.433,*layer3.1.conv1*:0.433,*layer3.1.conv2*:0.433,*layer3.2.conv1*:0.433,*layer3.2.conv2*:0.433,*layer3.3.conv1*:0.433,*layer3.3.conv2*:0.433,*layer3.4.conv1*:0.433,*layer3.4.conv2*:0.433,*layer3.5.conv1*:0.433,*layer3.5.conv2*:0.433,*layer4.0.conv1*:0.4316,*layer4.0.conv2*:0.4316,*layer4.1.conv1*:0.4316,*layer4.1.conv2*:0.4316,*layer4.2.conv1*:0.4316,*layer4.2.conv2*:0.4316 \
    --lr_ft 0:0.08,3:0.008,6:0.0008 --epochs 9 \
    --weight_decay 0.0001 \
    --batch_oracle 0 \
    --random yes \
    --save_combination yes \
    --num_missions 20 \
    --experiment_name oracle__resnet50__imagenet__conv10.5__lr2__savecombination

# Multi-GPU
for i in 2
do
    CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
        -a resnet50 --dataset imagenet \
        --data_path /datasets/fengsicheng/ \
        --pruner oracle --prune_method oracle --prune_with_hooks \
        --batch_size 256 \
        --stage_pr *conv1:0.4375,*layer1.0.conv1*:0.4375,*layer1.0.conv2*:0.4375,*layer1.1.conv1*:0.4375,*layer1.1.conv2*:0.4375,*layer1.2.conv1*:0.4375,*layer1.2.conv2*:0.4375,*layer2.0.conv1*:0.4375,*layer2.0.conv2*:0.4375,*layer2.1.conv1*:0.4375,*layer2.1.conv2*:0.4375,*layer2.2.conv1*:0.4375,*layer2.2.conv2*:0.4375,*layer2.3.conv1*:0.4375,*layer2.3.conv2*:0.4375,*layer3.0.conv1*:0.433,*layer3.0.conv2*:0.433,*layer3.1.conv1*:0.433,*layer3.1.conv2*:0.433,*layer3.2.conv1*:0.433,*layer3.2.conv2*:0.433,*layer3.3.conv1*:0.433,*layer3.3.conv2*:0.433,*layer3.4.conv1*:0.433,*layer3.4.conv2*:0.433,*layer3.5.conv1*:0.433,*layer3.5.conv2*:0.433,*layer4.0.conv1*:0.4316,*layer4.0.conv2*:0.4316,*layer4.1.conv1*:0.4316,*layer4.1.conv2*:0.4316,*layer4.2.conv1*:0.4316,*layer4.2.conv2*:0.4316 \
        --lr_ft 0:0.08,3:0.008,6:0.0008 --epochs 9 \
        --weight_decay 0.0001 \
        --batch_oracle $i \
        --random yes \
        --save_combination no \
        --num_missions 20 \
        --num_batch 8 \
        --experiment_name oracle__resnet50__imagenet__conv10.5__lr2__t1\_batch$i > ./record/resnet50_conv10.5_lr2_t1\_batch$i.txt &
done

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp --test_pretrained \
    -a resnet50 --dataset imagenet \
    --data_path /datasets/fengsicheng/ \
    --pruner l1 --prune_method l1 --prune_with_hooks \
    --batch_size 256 \
    --stage_pr *conv1:0.4375,*layer1.0.conv1*:0.4375,*layer1.0.conv2*:0.4375,*layer1.1.conv1*:0.4375,*layer1.1.conv2*:0.4375,*layer1.2.conv1*:0.4375,*layer1.2.conv2*:0.4375,*layer2.0.conv1*:0.4375,*layer2.0.conv2*:0.4375,*layer2.1.conv1*:0.4375,*layer2.1.conv2*:0.4375,*layer2.2.conv1*:0.4375,*layer2.2.conv2*:0.4375,*layer2.3.conv1*:0.4375,*layer2.3.conv2*:0.4375,*layer3.0.conv1*:0.433,*layer3.0.conv2*:0.433,*layer3.1.conv1*:0.433,*layer3.1.conv2*:0.433,*layer3.2.conv1*:0.433,*layer3.2.conv2*:0.433,*layer3.3.conv1*:0.433,*layer3.3.conv2*:0.433,*layer3.4.conv1*:0.433,*layer3.4.conv2*:0.433,*layer3.5.conv1*:0.433,*layer3.5.conv2*:0.433,*layer4.0.conv1*:0.4316,*layer4.0.conv2*:0.4316,*layer4.1.conv1*:0.4316,*layer4.1.conv2*:0.4316,*layer4.2.conv1*:0.4316,*layer4.2.conv2*:0.4316 \
    --lr_ft 0:0.08,30:0.008,60:0.0008 --epochs 90 \
    --weight_decay 0.0001 \
    --experiment_name l1__resnet50__imagenet__conv10.5__lr2__t1 > ./record/resnet50_conv10.5_lr2_l1_t1.txt


