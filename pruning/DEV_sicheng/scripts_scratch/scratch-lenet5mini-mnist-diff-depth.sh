export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=0 python main.py --no_ddp --no_scp \
    -a lenet5_mini_d6 --dataset mnist \
    --lr 0:0.01,20:0.001 --epochs 30 \
    --weight_decay 1e-4 \
    --experiment_name scratch__lenet5_mini_d6__mnist__epochs30_lr0.01to0.001


CUDA_VISIBLE_DEVICES=1 python main.py --no_ddp --no_scp \
    -a lenet5_mini_d7 --dataset mnist \
    --lr 0:0.01,20:0.001 --epochs 30 \
    --weight_decay 1e-4 \
    --experiment_name scratch__lenet5_mini_d7__mnist__epochs30_lr0.01to0.001


CUDA_VISIBLE_DEVICES=2 python main.py --no_ddp --no_scp \
    -a lenet5_mini_d8 --dataset mnist \
    --lr 0:0.01,20:0.001 --epochs 30 \
    --weight_decay 1e-4 \
    --experiment_name scratch__lenet5_mini_d8__mnist__epochs30_lr0.01to0.001