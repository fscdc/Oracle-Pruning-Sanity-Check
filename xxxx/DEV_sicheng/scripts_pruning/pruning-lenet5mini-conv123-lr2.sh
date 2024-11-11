#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

trap terminate SIGINT SIGTERM

if [ ! -d "./record" ]; then
    mkdir -p ./record
fi

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp \
    -a lenet5_mini --dataset mnist \
    --pretrained_ckpt ./Experiments/scratch__lenet5_mini__mnist__epochs30_lr0.01to0.001_SERVER063-20240711-224104/weights/ckpt_best.pth \
    --pruner oracle --prune_method oracle --prune_with_hooks \
    --stage_pr *conv1*:0.5,*conv2:0.5,*conv3:0.5 \
    --lr_ft 0:0.001,20:1e-4 --epochs 30 \
    --weight_decay 1e-4 \
    --batch_oracle 0 \
    --random yes \
    --save_combination yes \
    --experiment_name oracle__lenet5_mini__mnist__prconv1230.5__epoch30_lr0.001to0.0001

# repeat the same experiment for 3 times
for t in {1..3}
do
    # multi gpu
    for i in {0..7}
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
            -a lenet5_mini --dataset mnist \
            --pretrained_ckpt ./Experiments/scratch__lenet5_mini__mnist__epochs30_lr0.01to0.001_SERVER063-20240711-224104/weights/ckpt_best.pth \
            --pruner oracle --prune_method oracle --prune_with_hooks \
            --stage_pr *conv1*:0.5,*conv2:0.5,*conv3:0.5 \
            --lr_ft 0:0.001,20:1e-4 --epochs 30 \
            --weight_decay 1e-4 \
            --batch_oracle $i \
            --random yes \
            --save_combination no \
            --experiment_name oracle__lenet5_mini__mnist__prconv1230.5__epoch30_lr0.001to0.0001 > ./record/conv1230.5_lr2_t$t\_batch$i.txt &
    done
    wait

    CUDA_VISIBLE_DEVICES=6 python main.py --no_ddp --no_scp \
        -a lenet5_mini --dataset mnist \
        --pretrained_ckpt ./Experiments/scratch__lenet5_mini__mnist__epochs30_lr0.01to0.001_SERVER063-20240711-224104/weights/ckpt_best.pth \
        --pruner l1 --prune_method l1 --prune_with_hooks \
        --stage_pr *conv1*:0.5,*conv2:0.5,*conv3:0.5 \
        --lr_ft 0:0.001,20:0.0001 --epochs 30 \
        --weight_decay 1e-4 \
        --experiment_name l1__lenet5_mini__mnist__prconv1230.5 > ./record/conv1230.5_lr2_l1_t$t.txt &    
done