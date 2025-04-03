#!/bin/bash
# Default values
DEFAULT_IMAGE_FOLDER="/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
DEFAULT_JSON_FOLDER="/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json"
DEFAULT_LLM_FOLDER="/root/model/models--Qwen--Qwen-1_8B/snapshots/fa6e214ccbbc6a55235c26ef406355b6bfdf5eed"
DEFAULT_OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"
DEFAULT_OUTPUT_CKPT_NAME="llavaqwen-1.8b-pretrain"
DEFAULT_LOCALHOST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
DEFAULT_IMAGE_TOWER="openai/clip-vit-large-patch14-336 "
declare -a DEFAULT_DATA_PATHS=(
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_.json"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        --image-tower)
            IMAGE_TOWER="$2"
            shift 2
            ;;
        --localhost)
            LOCALHOST="$2"
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
        --llm-folder)
            LLM_FOLDER="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --output-ckpt-name)
            OUTPUT_CKPT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --image-folder PATH      Set image folder path (default: $DEFAULT_IMAGE_FOLDER)"
            echo "  --json-folder PATH       Set JSON folder path (default: $DEFAULT_JSON_FOLDER)"
            echo "  --llm-folder PATH        Set LLM folder path (default: $DEFAULT_LLM_FOLDER)"
            echo "  --outdir PATH            Set output directory (default: $DEFAULT_OUTDIR)"
            echo "  --output-ckpt-name NAME  Set output checkpoint name (default: $DEFAULT_OUTPUT_CKPT_NAME)"
            echo "  --help                   Display this help message"
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
DATA_PATHS_STRING="${DATA_PATHS[*]}"
IMAGE_TOWER=${IMAGE_TOWER:-$DEFAULT_IMAGE_TOWER}

LLM_FOLDER=${LLM_FOLDER:-$DEFAULT_LLM_FOLDER}
OUTDIR=${OUTDIR:-$DEFAULT_OUTDIR}
OUTPUT_CKPT_NAME=${OUTPUT_CKPT_NAME:-$DEFAULT_OUTPUT_CKPT_NAME}

# Construct the deepspeed include option
LOCALHOST=${LOCALHOST:-$DEFAULT_LOCALHOST}
DEEPSPEED_INCLUDE="--include localhost:$LOCALHOST"

cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    $DEEPSPEED_INCLUDE \
    moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2_debug.json \
    --model_name_or_path ${LLM_FOLDER} \
    --version plain \
    --data_path ${DATA_PATHS_STRING} \
    --image_tower ${IMAGE_TOWER} \
    --image_folder ${IMAGE_FOLDER} \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTDIR/$OUTPUT_CKPT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
