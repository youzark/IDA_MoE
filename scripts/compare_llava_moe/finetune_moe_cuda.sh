#!/bin/bash
# Default values
export HCCL_ALGO="level0:NA;level1:H-D_R"
DEFAULT_IMAGE_FOLDER="/root/data/LLaVA-Finetune"
declare -a DEFAULT_DATA_PATHS=(
    "/root/data/LLaVA-Finetune/llava_v1_5_mix665k.json"
)
DEFAULT_START_CKPT_NAME="llavaqwen-1.8b-finetune"
DEFAULT_OUTPUT_CKPT_NAME="llavaqwen-1.8b-finetune-moe_v3"
DEFAULT_OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"
DEFAULT_MOE_MODE="sparse"
DEFAULT_NUM_EXPERTS=4
DEFAULT_TOP_K_EXPERTS=2
DEFAULT_USE_RESIDUAL="False"
DEFAULT_ROUTER_AUX_LOSS_COEF=0.01
DEFAULT_L_AUX_TYPE="load_balancing"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        --data-paths)
            # Read the paths into an array
            shift
            DATA_PATHS=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATA_PATHS+=("$1")
                shift
            done
            ;;
        --start-ckpt-name)
            START_CKPT_NAME="$2"
            shift 2
            ;;
        --output-ckpt-name)
            OUTPUT_CKPT_NAME="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --moe-mode)
            moe_mode="$2"
            shift 2
            ;;
        --num-experts)
            num_experts="$2"
            shift 2
            ;;
        --top-k-experts)
            top_k_experts="$2"
            shift 2
            ;;
        --use-residual)
            use_residual="$2"
            shift 2
            ;;
        --router-aux-loss-coef)
            router_aux_loss_coef="$2"
            shift 2
            ;;
        --l-aux-type)
            l_aux_type="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --image-folder PATH            Set image folder path (default: $DEFAULT_IMAGE_FOLDER)"
            echo "  --data-paths PATH1 [PATH2...]  Set data paths (multiple paths allowed)"
            echo "  --start-ckpt-name NAME         Set start checkpoint name (default: $DEFAULT_START_CKPT_NAME)"
            echo "  --output-ckpt-name NAME        Set output checkpoint name (default: $DEFAULT_OUTPUT_CKPT_NAME)"
            echo "  --outdir PATH                  Set output directory (default: $DEFAULT_OUTDIR)"
            echo "  --moe-mode MODE                Set MoE mode (default: $DEFAULT_MOE_MODE)"
            echo "  --num-experts NUM              Set number of experts (default: $DEFAULT_NUM_EXPERTS)"
            echo "  --top-k-experts NUM            Set top-k experts (default: $DEFAULT_TOP_K_EXPERTS)"
            echo "  --use-residual BOOL            Set use residual (default: $DEFAULT_USE_RESIDUAL)"
            echo "  --router-aux-loss-coef NUM     Set router aux loss coefficient (default: $DEFAULT_ROUTER_AUX_LOSS_COEF)"
            echo "  --l-aux-type TYPE              Set l aux type (default: $DEFAULT_L_AUX_TYPE)"
            echo "  --help                         Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
IMAGE_FOLDER=${IMAGE_FOLDER:-$DEFAULT_IMAGE_FOLDER}
DATA_PATHS=("${DATA_PATHS[@]:-${DEFAULT_DATA_PATHS[@]}}")
START_CKPT_NAME=${START_CKPT_NAME:-$DEFAULT_START_CKPT_NAME}
OUTPUT_CKPT_NAME=${OUTPUT_CKPT_NAME:-$DEFAULT_OUTPUT_CKPT_NAME}
OUTDIR=${OUTDIR:-$DEFAULT_OUTDIR}
moe_mode=${moe_mode:-$DEFAULT_MOE_MODE}
num_experts=${num_experts:-$DEFAULT_NUM_EXPERTS}
top_k_experts=${top_k_experts:-$DEFAULT_TOP_K_EXPERTS}
use_residual=${use_residual:-$DEFAULT_USE_RESIDUAL}
router_aux_loss_coef=${router_aux_loss_coef:-$DEFAULT_ROUTER_AUX_LOSS_COEF}
l_aux_type=${l_aux_type:-$DEFAULT_L_AUX_TYPE}

# Join array elements with spaces for the command
DATA_PATHS_STRING="${DATA_PATHS[*]}"

# Echo settings summary
echo "=== Running with the following settings ==="
echo "Image Folder:        ${IMAGE_FOLDER}"
echo "Data Paths:"
for path in "${DATA_PATHS[@]}"; do
    echo "  - ${path}"
done
echo "Start Checkpoint:    ${START_CKPT_NAME}"
echo "Output Checkpoint:   ${OUTPUT_CKPT_NAME}"
echo "Output Directory:    ${OUTDIR}"
echo "MoE Settings:"
echo "  Mode:             ${moe_mode}"
echo "  Num Experts:      ${num_experts}"
echo "  Top-K Experts:    ${top_k_experts}"
echo "  Use Residual:     ${use_residual}"
echo "  Aux Loss Coef:    ${router_aux_loss_coef}"
echo "  L Aux Type:       ${l_aux_type}"

cd /apdcephfs_nj7_1273717/share_1273717/yannhua/Home/moe
deepspeed \
    --master_port 29500 \
    --include localhost:0,1,2,3,4,5,6,7 \
    moellava/train/train_mem_cuda.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} --l_aux_type ${l_aux_type} \
    --train_modules w1 w2 c_proj wg \
    --deepspeed ./scripts/zero2_debug.json \
    --model_name_or_path ${OUTDIR}/${START_CKPT_NAME} \
    --version qwen \
    --data_path ${DATA_PATHS_STRING} \
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
    --save_steps 5000 \
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
