#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

save_dir='xxx' # specify your save path
cfg=changedetection/configs/vssm1/vssm_small_224.yaml
pretrain_weight=None
model_type=FlowMamba
ckpt='xxx.pth' # specify your checkpoint path
batch_size=1
crop_size=256
max_iters=800000
model_save_path='xxx' # specify your model save path
log_file=$model_save_path/test.log
data_name=xBD
data_root='xxx' # specify your dataset path
test_data=$data_root/test
test_data_list=$data_root/test/image_list.txt

python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        changedetection/script/test.py \
        --cfg $cfg \
        --pretrained_weight_path $pretrain_weight \
        --dataset $data_name \
        --test_dataset_path $test_data \
        --test_data_list_path $test_data_list \
        --batch_size $batch_size \
        --crop_size $crop_size \
        --model_type $model_type \
        --result_saved_path $model_save_path \
        --log_file $log_file \
        --resume $ckpt \