export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=1 python main.py --no_ddp --no_scp \
    --arch resnet56 \
    --dataset cifar10 \
    --batch_size 128 \
    --weight_decay 0.0005 \
    --lr 0:0.1,100:0.01,150:0.001 \
    --epochs 200 \
    --experiment_name scratch__resnet56__cifar10