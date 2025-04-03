#!/bin/bash

function train_on_dataset() {
    local json_path=$1        # Full path to json file
    local prev_ckpt=$2       # Name of the previous checkpoint to start from
    local dataset_tag=$3     # Short identifier for the dataset
    local base_model=$4      # Base model name
    local l_aux_type=$5
    local router_aux_loss_coef=$6
    
    # Extract base directory from json path
    local dataset_path=$(dirname "${json_path}")
    local json_name=$(basename "${json_path}")

    # MOE parameters
    local moe_mode="sparse"
    local num_experts=4
    local top_k_experts=2
    local use_residual=False

    # Construct checkpoint paths
    local base_dir="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/${base_model}"
    local model_dir="${base_dir}/${moe_tag}"
    local output_ckpt="${prev_ckpt}_${dataset_tag}"
    
    mkdir -p ${model_dir}
    
    export HCCL_ALGO="level0:NA;level1:H-D_R"
    cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
        --master_port 29500 \
        --include localhost:0,1,2,3,4,5,6,7 \
        moellava/train/train_mem.py \
        --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
        --moe_mode ${moe_mode} --use_residual ${use_residual} \
        --router_aux_loss_coef ${router_aux_loss_coef} --l_aux_type ${l_aux_type} \
        --train_modules w1 w2 c_proj wg \
        --deepspeed ./scripts/zero2_debug.json \
        --model_name_or_path ${model_dir}/${prev_ckpt} \
        --version qwen \
        --data_path ${json_path} \
        --image_folder ${dataset_path} \
        --image_tower openai/clip-vit-large-patch14-336 \
        --image_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${model_dir}/${output_ckpt} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
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
}

# Dataset JSON paths
DATASETS=(
    "/root/data/ChEBI-20_data/train_file.json"
    "/root/data/LLaVA-Math/datasets--Zhiqiang007--MathV360K/snapshots/a3eb6686c97c3234ed50487d5002618404c122a3/train_samples_all_tuning.json"
    "/root/data/LLaVA-Art/train_samples_tuning.json"
)

# Dataset tags (short names for checkpoints)
TAGS=(
    "chem"
    "math"
    "art"
)

# Base configuration
BASE_MODEL="llavaqwen-1.8b-finetune"
LOSS_TYPE="load_balancing"
LOSS_COEF=0.01

# Convert float and handle routing type
coef=$(printf "%03d" $(echo "${LOSS_COEF} * 100" | bc | cut -d. -f1)) || exit 1

[[ ${LOSS_TYPE} == "load_balancing" ]] && lb_or_sp="lb" || lb_or_sp="sp"
moe_tag="${lb_or_sp}${coef}"
    

# Sequential training
PREV_CKPT="${BASE_MODEL}_${moe_tag}"
for i in "${!DATASETS[@]}"; do
    train_on_dataset \
        "${DATASETS[$i]}" \
        "${PREV_CKPT}" \
        "${TAGS[$i]}" \
        "${BASE_MODEL}" \
        "$LOSS_TYPE" \
        $LOSS_COEF

    PREV_CKPT=${PREV_CKPT}_${TAGS[$i]}
done
