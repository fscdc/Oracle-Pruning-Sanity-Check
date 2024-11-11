export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=1 python main.py --no_ddp --no_scp \
    -a lenet5_mini --dataset kmnist \
    --lr 0:0.01,20:0.001 --epochs 30 \
    --weight_decay 1e-4 \
    --experiment_name scratch__lenet5_mini__kmnist__epochs30_lr0.01to0.001
