#!/bin/bash

# Default values
export HCCL_ALGO="level0:NA;level1:H-D_R"
DEFAULT_LLM_FOLDER="/root/model/models--Qwen--Qwen-1_8B/snapshots/fa6e214ccbbc6a55235c26ef406355b6bfdf5eed"
DEFAULT_IMAGE_FOLDER="/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
declare -a DEFAULT_DATA_PATHS=(
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json"
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json"
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json"
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json"
    "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json"
)
DEFAULT_OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"
DEFAULT_START_CKPT_NAME="llavaqwen-1.8b-pretrain"
DEFAULT_OUTPUT_CKPT_NAME="llavaqwen-1.8b-finetune-org"
DEFAULT_LOCALHOST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
DEFAULT_IMAGE_TOWER="openai/clip-vit-large-patch14-336 "
DEFAULT_BATCH_SIZE=4
DEFAULT_VERSION="qwen"
DEFAULT_EPOCH=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm-folder)
            LLM_FOLDER="$2"
            shift 2
            ;;
        --image-tower)
            IMAGE_TOWER="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --start-ckpt-name)
            START_CKPT_NAME="$2"
            shift 2
            ;;
        --output-ckpt-name)
            OUTPUT_CKPT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --llm-folder PATH              Set LLM folder path (default: $DEFAULT_LLM_FOLDER)"
            echo "  --image-folder PATH            Set image folder path (default: $DEFAULT_IMAGE_FOLDER)"
            echo "  --data-paths PATH1 [PATH2...]  Set data paths (multiple paths allowed)"
            echo "  --outdir PATH                  Set output directory (default: $DEFAULT_OUTDIR)"
            echo "  --start-ckpt-name NAME         Set start checkpoint name (default: $DEFAULT_START_CKPT_NAME)"
            echo "  --output-ckpt-name NAME        Set output checkpoint name (default: $DEFAULT_OUTPUT_CKPT_NAME)"
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
LLM_FOLDER=${LLM_FOLDER:-$DEFAULT_LLM_FOLDER}
IMAGE_FOLDER=${IMAGE_FOLDER:-$DEFAULT_IMAGE_FOLDER}
IMAGE_TOWER=${IMAGE_TOWER:-$DEFAULT_IMAGE_TOWER}
DATA_PATHS=("${DATA_PATHS[@]:-${DEFAULT_DATA_PATHS[@]}}")
OUTDIR=${OUTDIR:-$DEFAULT_OUTDIR}
START_CKPT_NAME=${START_CKPT_NAME:-$DEFAULT_START_CKPT_NAME}
OUTPUT_CKPT_NAME=${OUTPUT_CKPT_NAME:-$DEFAULT_OUTPUT_CKPT_NAME}
VERSION=${VERSION:-$DEFAULT_VERSION}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

# Join array elements with spaces for the command
DATA_PATHS_STRING="${DATA_PATHS[*]}"

# Construct the deepspeed include option
LOCALHOST=${LOCALHOST:-$DEFAULT_LOCALHOST}
DEEPSPEED_INCLUDE="--include localhost:$LOCALHOST"
EPOCH=${EPOCH:-$DEFAULT_EPOCH}

echo "=== Running with the following settings ==="
echo "LLM Folder:          ${LLM_FOLDER}"
echo "Image Folder:        ${IMAGE_FOLDER}"
echo "Output Directory:    ${OUTDIR}"
echo "Start Checkpoint:    ${START_CKPT_NAME}"
echo "Output Checkpoint:   ${OUTPUT_CKPT_NAME}"
echo "Data Paths:"
for path in "${DATA_PATHS[@]}"; do
    echo "  - ${path}"
done

cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    $DEEPSPEED_INCLUDE \
    moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2_debug.json \
    --model_name_or_path ${LLM_FOLDER} \
    --version $VERSION \
    --data_path ${DATA_PATHS_STRING} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER} \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter $OUTDIR/${START_CKPT_NAME}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTDIR/${OUTPUT_CKPT_NAME} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
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
