#!/bin/bash

MASTER_PORT=10078
lr=1e-4
wd=1e-4
# Model and Training Parameters
n_gpu=$(nvidia-smi -L | wc -l)
update_freq=1
x_norm_loss=0.01
delta_pair_repr_norm_loss=-1
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=0.8
seed=42
warmup_steps=100
max_steps=1000000
batch_size=16
run_name="pretrain"
save_dir=./results/pretrain

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1


# Training Command
echo "Starting training..."
echo "GPUs: $n_gpu"
echo "Global batch size: $batch_size"
echo "Save directory: $save_dir"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$n_gpu --nnodes=1 --node-rank=0 --master_port=$MASTER_PORT $(which unicore-train) ./data  --user-dir ./unimol --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d \
    --task solvclip --loss solvclip --arch solvclip  \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
    --update-freq $update_freq --seed $seed \
    --tensorboard-logdir $save_dir/tsb \
    --max-update $max_steps --log-interval 10 --log-format simple \
    --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
    --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
    --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
    --save-dir $save_dir  --only-polar $only_polar --run-name $run_name \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --lig-pretrained ./mol_pre_no_h_220816.pt \
    --pocket-pretrained ./poc_pre_220816.pt \
    --mask-feature 1 \
