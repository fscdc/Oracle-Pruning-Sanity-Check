#!/bin/bash

# define an array of combinations
declare -a combinations=(
    "conv102_lr2 0.035 0.05 0.98 0.995 0.0 7.3"
    "conv103_lr2 0.03 0.07 0.975 0.995 0.0 10"
    "conv104_lr2 0.03 0.08 0.975 0.995 0.0 10"
    "conv105_lr2 0.03 0.08 0.975 0.995 0.0 11"
    "conv106_lr2 0.03 0.08 0.975 0.99 0.0 11"
    "conv107_lr2 0.03 0.09 0.97 0.99 0.0 14"
    "conv108_lr2 0.04 0.1 0.965 0.985 0.0 14"
    "conv202_lr2 0.03 0.07 0.975 0.99 0.0 10"
    "conv203_lr2 0.03 0.08 0.97 0.99 0.0 10"
    "conv204_lr2 0.03 0.08 0.97 0.99 0.0 10"
    "conv205_lr2 0.03 0.1 0.965 0.99 0.0 10"
    "conv206_lr2 0.04 0.1 0.965 0.99 0.0 8"
    "conv207_lr2 0.04 0.12 0.96 0.985 1.5 7"
    "conv208_lr2 0.075 0.2 0.93 0.98 2 8"
    "conv302_lr2 0.03 0.09 0.97 0.99 0.0 14"
    "conv303_lr2 0.04 0.09 0.97 0.99 0.5 15"
    "conv304_lr2 0.04 0.14 0.96 0.99 2 18"
    "conv305_lr2 0.05 0.25 0.93 0.99 3 18.5"
    "conv306_lr2 0.07 0.28 0.92 0.98 4 16"
    "conv307_lr2 0.1 0.55 0.86 0.96 3 13"
    "conv308_lr2 0.3 0.9 0.78 0.92 2.5 11"
    "conv10.520.5_lr2 0.03 0.12 0.96 0.99 1.5 10"
    "conv10.530.5_lr2 0.0 0.6 0.85 0.99 2 15"
    "conv20.530.5_lr2 0.07 0.17 0.94 0.98 2.5 12 "
    "conv1230.5_lr2 0.01 0.6 0.93 0.98 1.5 10"
    "vgg19_conv0.5_lr2 1.55 1.75 0.65 0.68 4.8 5.6"
    "resnet56_conv10.5_lr2 0.28 0.38 0.9 0.93 0.0 35.0"
    "resnet18_conv10.5_lr2 1.5 1.65 0.58 0.66 0.0 120.0"
    "fmnist_conv105_lr2 0.3 0.45 0.85 0.90 0.0 8"
    "kmnist_conv105_lr2 0.3 0.55 0.84 0.92 0.0 8"
    "d6_conv105_lr2 0.03 0.08 0.975 0.995 0.0 12"
    "d7_conv105_lr2 0.03 0.08 0.975 0.995 0.0 12"
    "d8_conv105_lr2 0.05 0.12 0.965 0.99 2 9"
    "w20_conv105_lr2 0.03 0.06 0.98 0.99 0.0 8"
    "w30_conv105_lr2 0.03 0.06 0.98 0.99 0.0 10"
    "w40_conv105_lr2 0.03 0.06 0.98 0.995 0.0 8"
)

# iterate through the array and execute the commands
for combo in "${combinations[@]}"
do
    set -- $combo
    filename=$1
    y_min=$2
    y_max=$3
    test_acc_y_min=$4
    test_acc_y_max=$5
    x_min=$6
    x_max=$7

    python log_preprocess.py --filename "$filename"

    python analysis.py --filename "$filename" --test_acc_y_min "$test_acc_y_min" --test_acc_y_max "$test_acc_y_max"  --y_max "$y_max" --y_min "$y_min" --x_min "$x_min" --x_max "$x_max"

    python log_preprocess_final.py --filename "$filename"
done
