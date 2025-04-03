#!/bin/bash

# Default values
IS_MOE=false
BASE_MODEL="llavaqwen-1.8b-finetune"
LOSS_TYPE="load_balancing"
LOSS_COEF=0.01

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --is-moe)
            IS_MOE="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --loss-type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --loss-coef)
            LOSS_COEF="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --is-moe     Enable MOE training (default: false)"
            echo "  --base-model Base model name (default: llavaqwen-1.8b-finetune)"
            echo "  --loss-type  Loss type for MOE (default: load_balancing)"
            echo "  --loss-coef  Loss coefficient (default: 0.01)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

function train_on_dataset() {
    local json_path=$1        # Full path to json file
    local prev_ckpt=$2       # Name of the previous checkpoint to start from
    local dataset_tag=$3     # Short identifier for the dataset
    local base_model=$4      # Base model name
    local is_moe=$5         # Boolean for MOE or dense training
    local l_aux_type=$6      # Only used if is_moe=true
    local router_aux_loss_coef=$7  # Only used if is_moe=true
    
    # Extract base directory from json path
    local dataset_path=$(dirname "${json_path}")
    local json_name=$(basename "${json_path}")

    # Base directory setup
    local base_dir="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps"
    local model_dir
    local output_ckpt

    # MOE-specific parameters
    local moe_args=""
    if [ "$is_moe" = true ]; then
        local moe_mode="sparse"
        local num_experts=4
        local top_k_experts=2
        local use_residual=False
        
        model_dir="${base_dir}/${base_model}/${moe_tag}"
        output_ckpt="${prev_ckpt}_${dataset_tag}"
        
        moe_args="--moe_enable True \
                 --num_experts ${num_experts} \
                 --top_k_experts ${top_k_experts} \
                 --capacity_factor 1.5 \
                 --moe_mode ${moe_mode} \
                 --use_residual ${use_residual} \
                 --router_aux_loss_coef ${router_aux_loss_coef} \
                 --l_aux_type ${l_aux_type} \
                 --train_modules w1 w2 c_proj wg"
    else
        model_dir="${base_dir}/${base_model}/dense"
        output_ckpt="${prev_ckpt}_${dataset_tag}"
    fi
    
    mkdir -p ${model_dir}
    
    export HCCL_ALGO="level0:NA;level1:H-D_R"
    cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
    
    # Determine batch size based on model type
    local batch_size
    if [ "$is_moe" = true ]; then
        batch_size=4
    else
        batch_size=16
    fi
    
    # Common arguments for both MOE and dense training
    local common_args="--deepspeed ./scripts/zero2_debug.json \
        --version qwen \
        --image_tower openai/clip-vit-large-patch14-336 \
        --image_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --num_train_epochs 1 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --per_device_eval_batch_size 4 \
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
        --cache_dir "./cache_dir""

    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
        --master_port 29500 \
        --include localhost:0,1,2,3,4,5,6,7 \
        moellava/train/train_mem.py \
        ${moe_args} \
        --model_name_or_path ${model_dir}/${prev_ckpt} \
        --data_path ${json_path} \
        --image_folder ${dataset_path} \
        --output_dir ${model_dir}/${output_ckpt} \
        --per_device_train_batch_size ${batch_size} \
        ${common_args} 
}

# Dataset configurations
DATASETS=(
    "/root/data/ChEBI-20_data/train_file.json"
    "/root/data/LLaVA-Math/datasets--Zhiqiang007--MathV360K/snapshots/a3eb6686c97c3234ed50487d5002618404c122a3/train_samples_all_tuning.json"
    "/root/data/LLaVA-Art/train_samples_tuning.json"
)

TAGS=(
    "chem"
    "math"
    "art"
)

# Convert float and handle routing type for MOE
coef=$(printf "%03d" $(echo "${LOSS_COEF} * 100" | bc | cut -d. -f1)) || exit 1
[[ ${LOSS_TYPE} == "load_balancing" ]] && lb_or_sp="lb" || lb_or_sp="sp"
moe_tag="${lb_or_sp}${coef}"

# Train both MOE and dense models
if [ "$IS_MOE" = true ]; then
    BASE_MODEL="${BASE_MODEL}-moe"
    PREV_CKPT="${BASE_MODEL}_${moe_tag}"
else
    moe_tag=dense
    PREV_CKPT="${BASE_MODEL}_${moe_tag}"
    BASE_MODEL="${BASE_MODEL}-moe"
fi

for i in "${!DATASETS[@]}"; do
    train_on_dataset \
        "${DATASETS[$i]}" \
        "${PREV_CKPT}" \
        "${TAGS[$i]}" \
        "${BASE_MODEL}" \
        $IS_MOE \
        "$LOSS_TYPE" \
        $LOSS_COEF

    PREV_CKPT=${PREV_CKPT}_${TAGS[$i]}
done
