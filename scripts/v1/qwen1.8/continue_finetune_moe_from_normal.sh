#!/bin/bash

export HCCL_ALGO="level0:NA;level1:H-D_R"
moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
IMAGE_FOLDER="/root/data/LLaVA-Math/datasets--Zhiqiang007--MathV360K/snapshots/a3eb6686c97c3234ed50487d5002618404c122a3/data_images"
JSON_FOLDER="/root/data/LLaVA-Math/datasets--Zhiqiang007--MathV360K/snapshots/a3eb6686c97c3234ed50487d5002618404c122a3"

START_CKPT_NAME="llavaqwen-1.8b-finetune"
CKPT_NAME="llavaqwen-1.8b-finetune-moe-math-normal"
OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints"
cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    --include localhost:0,1,2,3,4,5,6,7 \
    moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2_debug.json \
    --model_name_or_path ./checkpoints/${START_CKPT_NAME} \
    --version qwen \
    --data_path ${JSON_FOLDER}/train_samples_all_tuning.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/${CKPT_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"






