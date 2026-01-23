#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export TORCH_DISTRIBUTED_DEBUG=INFO
GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29504}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

save_dir='xxx' # change this to your save path
batch_size=8
crop_size=256
max_iters=80000
model_type=FlowMamba
cfg=changedetection/configs/vssm1/vssm_small_224.yaml
model_save_path=$save_dir # change this to your save path
pretrained_path=pretrained/vssm_small_0229_ckpt_epoch_222.pth
log_file=$model_save_path/train_val.log
data_name=xBD # change this to your data name, e.g., xBD, bright
data_root='test' # change this to your data root path
train_data=$data_root/train
train_data_list=$train_data/image_list.txt
test_data=$data_root/test
test_data_list=$test_data/image_list.txt

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    changedetection/script/train.py \
    --dataset $data_name \
    --batch_size $batch_size \
    --crop_size $crop_size \
    --max_iters $max_iters \
    --model_type $model_type \
    --model_param_path $model_save_path \
    --pretrained_weight_path $pretrained_path \
    --log_file $log_file \
    --cfg $cfg \
    --train_dataset_path $train_data \
    --train_data_list_path $train_data_list \
    --test_dataset_path $test_data \
    --test_data_list_path $test_data_list
