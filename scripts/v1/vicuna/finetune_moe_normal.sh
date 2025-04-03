#!/bin/bash
#
export HCCL_ALGO="level0:NA;level1:H-D_R"
IMAGE_FOLDER="/root/data/LLaVA-Finetune"
JSON_FOLDER="/root/data/LLaVA-Finetune"
START_CKPT_NAME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints/llava-v1.5-7b-finetune"
OUTPUT_CKPT_NAME="llava-v1.5-7b-finetune-moe-normal-0.01"
OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
l_aux_type="load_balancing"

cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    --include localhost:0,1,2,3,4,5,6,7 \
    moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} --l_aux_type ${l_aux_type} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2_debug.json \
    --model_name_or_path ${START_CKPT_NAME} \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_v1_5_mix665k.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTDIR}/${OUTPUT_CKPT_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
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
