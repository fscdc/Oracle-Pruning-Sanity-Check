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

# Define a list of stage pruning rates
declare -a stage_pr_rates=("0.5")

# Repeat the same experiment for different stage pruning rates and for 3 times each
for stage_pr in "${stage_pr_rates[@]}"
do
    # repeat the same experiment for 3 times
    for t in {1..3}
    do
        # multi gpu
        for i in {0..7}
        do
            CUDA_VISIBLE_DEVICES=$i python main.py --no_ddp --no_scp \
                -a lenet5_mini --dataset fmnist \
                --pretrained_ckpt ./Experiments/scratch__lenet5_mini__fmnist__epochs30_lr0.01to0.001_SERVER227-20241027-133154/weights/ckpt_best.pth \
                --pruner oracle --prune_method oracle --prune_with_hooks \
                --stage_pr "*conv1*:$stage_pr" \
                --lr_ft 0:0.001,20:1e-4 --epochs 30 \
                --weight_decay 1e-4 \
                --batch_oracle $i \
                --experiment_name oracle__lenet5_mini__fmnist__prconv1${stage_pr//./}__epoch30_lr0.001to0.0001 > ./record/fmnist_conv1${stage_pr//./}_lr2_t$t\_batch$i.txt &
        done
        wait
    done
done