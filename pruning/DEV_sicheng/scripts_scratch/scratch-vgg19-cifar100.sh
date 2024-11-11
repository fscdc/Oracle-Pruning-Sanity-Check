export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=1 python main.py --no_ddp --no_scp \
    -a vgg19_C --dataset cifar100 \
    --weight_decay 5e-4 \
    --lr 0:0.1,100:0.01,150:0.001 \
    --epochs 200 \
    --experiment_name scratch__vgg19__cifar100