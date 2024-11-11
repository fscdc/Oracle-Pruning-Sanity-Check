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

if [ ! -d "./record" ]; then
    mkdir -p ./record
fi

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp --test_pretrained \
    -a vgg19_C --dataset cifar100 \
    --pretrained_ckpt ./Experiments/scratch__vgg19__cifar100_SERVER230-20240725-234244/weights/ckpt_best.pth \
    --pruner oracle --prune_method oracle --prune_with_hooks \
    --batch_size 256 \
    --stage_pr *features.0*:0.5,*features.3:0.5,*features.7:0.5,*features.10:0.5,*features.14:0.5,*features.17:0.5,*features.20:0.5,*features.24:0.5,*features.27:0.5,*features.30:0.5,*features.34:0.5,*features.37:0.5,*features.40:0.5,*features.43:0.5,*features.46:0.5,*features.49:0.5 \
    --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
    --weight_decay 0.0005 \
    --batch_oracle 0 \
    --random yes \
    --save_combination yes \
    --experiment_name oracle__vgg19__cifar100__conv0.5__lr2

# repeat the same experiment for 3 times
for t in {1..3}
do
    # multi gpu
    for i in {0..7}
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
            -a vgg19_C --dataset cifar100 \
            --pretrained_ckpt ./Experiments/scratch__vgg19__cifar100_SERVER230-20240725-234244/weights/ckpt_best.pth \
            --pruner oracle --prune_method oracle --prune_with_hooks \
            --batch_size 256 \
            --stage_pr *features.0*:0.5,*features.3:0.5,*features.7:0.5,*features.10:0.5,*features.14:0.5,*features.17:0.5,*features.20:0.5,*features.24:0.5,*features.27:0.5,*features.30:0.5,*features.34:0.5,*features.37:0.5,*features.40:0.5,*features.43:0.5,*features.46:0.5,*features.49:0.5 \
            --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
            --weight_decay 0.0005 \
            --batch_oracle $i \
            --random yes \
            --save_combination no \
            --experiment_name oracle__vgg19__cifar100__conv0.5__lr2 > ./record/vgg19_conv0.5_lr2_t$t\_batch$i.txt &
    done
    wait

    CUDA_VISIBLE_DEVICES=6 python main.py --no_ddp --no_scp --test_pretrained \
        -a vgg19_C --dataset cifar100 \
        --pretrained_ckpt ./Experiments/scratch__vgg19__cifar100_SERVER230-20240725-234244/weights/ckpt_best.pth \
        --pruner l1 --prune_method l1 --prune_with_hooks \
        --batch_size 256 \
        --stage_pr *features.0*:0.5,*features.3:0.5,*features.7:0.5,*features.10:0.5,*features.14:0.5,*features.17:0.5,*features.20:0.5,*features.24:0.5,*features.27:0.5,*features.30:0.5,*features.34:0.5,*features.37:0.5,*features.40:0.5,*features.43:0.5,*features.46:0.5,*features.49:0.5 \
        --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 \
        --weight_decay 0.0005 \
        --experiment_name l1__vgg19__cifar100__conv0.5__lr2 > ./record/vgg19_conv0.5_lr2_l1_t$t.txt &    
done
