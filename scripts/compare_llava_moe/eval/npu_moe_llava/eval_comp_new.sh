#!/bin/bash
function get_available_port() {
    local BASE_PORT=29500
    local MAX_PORT=39500
    local PORT_FILE="/tmp/used_ports.txt"
    local LOCK_FILE="/tmp/ports.lock"

    # Create files if they don't exist
    touch $PORT_FILE
    touch $LOCK_FILE

    # Acquire lock using flock
    exec 200>$LOCK_FILE
    flock -x 200

    while true; do
        local PORT=$(shuf -i $BASE_PORT-$MAX_PORT -n 1)
        if ! grep -q "^${PORT}$" $PORT_FILE; then
            echo $PORT >> $PORT_FILE
            # Release lock
            flock -u 200
            echo $PORT
            return 0
        fi
    done
}

function release_port() {
    local PORT=$1
    local PORT_FILE="/tmp/used_ports.txt"
    local LOCK_FILE="/tmp/ports.lock"

    # Acquire lock using flock
    exec 200>$LOCK_FILE
    flock -x 200

    # Create temporary file
    local TEMP_FILE=$(mktemp)
    grep -v "^${PORT}$" $PORT_FILE > $TEMP_FILE
    mv $TEMP_FILE $PORT_FILE

    # Release lock
    flock -u 200
}

function run_inference() {
    local BASE=$1
    local CKPT_NAME=$2
    local EVAL_DATASET=$3
    local USE_MOE=$4  # New parameter for MoE flag
    local BASE_MASTER_PORT=$(get_available_port)
    local ORIGINAL_DIR=$(pwd)

    CONV="qwen"
    MOE_HOME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
    CKPT="${BASE}/${CKPT_NAME}"
    EVAL="/root/eval"
    NUM_CHUNKS=8

    # Dataset-specific configurations
    case $EVAL_DATASET in
        "pope")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_loader"
            QUESTION_FILE="${EVAL}/pope/llava_pope_test.jsonl"
            IMAGE_FOLDER="${EVAL}/pope/val2014"
            ANSWERS_DIR="${EVAL}/pope/answers/${CKPT_NAME}"
            ;;
        "mmbench")
            EVAL_SCRIPT="moellava/eval/model_vqa_mmbench.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_mmbench"
            SPLIT="mmbench_dev_20230712"
            QUESTION_FILE="${EVAL}/mmbench/${SPLIT}.tsv"
            ANSWERS_DIR="${EVAL}/mmbench/answers/${CKPT_NAME}"
            ;;
        "mme")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_loader"
            QUESTION_FILE="${EVAL}/MME/llava_mme.jsonl"
            IMAGE_FOLDER="${EVAL}/MME/MME_Benchmark_release_version"
            ANSWERS_DIR="${EVAL}/MME/answers/${CKPT_NAME}"
            mkdir -p ${EVAL}/MME/results/${CKPT_NAME}
            ;;
        "mmvet")
            EVAL_SCRIPT="moellava/eval/model_vqa.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa"
            QUESTION_FILE="${EVAL}/mm-vet/llava-mm-vet.jsonl"
            IMAGE_FOLDER="${EVAL}/mm-vet/images"
            ANSWERS_DIR="${EVAL}/mm-vet/answers/${CKPT_NAME}"
            ;;
        "gqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_loader"
            QUESTION_FILE="${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl"
            IMAGE_FOLDER="${EVAL}/gqa/data/images"
            ANSWERS_DIR="${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}"
            ;;
        "sqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_science.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_science"
            QUESTION_FILE="${EVAL}/scienceqa/llava_test_CQM-A.json"
            IMAGE_FOLDER="${EVAL}/scienceqa/images/test"
            ANSWERS_DIR="${EVAL}/scienceqa/answers/${CKPT_NAME}"
            ;;
        "textvqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_loader"
            QUESTION_FILE="${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl"
            IMAGE_FOLDER="${EVAL}/textvqa/train_images"
            ANSWERS_DIR="${EVAL}/textvqa/answers/${CKPT_NAME}"
            ;;
        "vizwiz")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            EVAL_DENSE_SCRIPT="moellava.eval.model_vqa_loader"
            QUESTION_FILE="${EVAL}/vizwiz/llava_test.jsonl"
            IMAGE_FOLDER="${EVAL}/vizwiz/test"
            ANSWERS_DIR="${EVAL}/vizwiz/answers/${CKPT_NAME}"
            ;;
        *)
            echo "Unsupported dataset: ${EVAL_DATASET}"
            return 1
            ;;
    esac

    mkdir -p ${ANSWERS_DIR}
    cd ${MOE_HOME}

    # All datasets use parallel processing
    if [[ "$USE_MOE" == "true" ]]; then  # Changed condition to use USE_MOE parameter
        for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
            MASTER_PORT=$(get_available_port)
            INCLUDE_DEVICES="localhost:${IDX}"

            # Additional args for SQA
            EXTRA_ARGS=""
            if [[ $EVAL_DATASET == "sqa" ]]; then
                EXTRA_ARGS="--single-pred-prompt"
            fi

            deepspeed \
                --master_port $MASTER_PORT \
                --include $INCLUDE_DEVICES \
                ${EVAL_SCRIPT} \
                --model-path ${CKPT} \
                --question-file ${QUESTION_FILE} \
                --image-folder ${IMAGE_FOLDER} \
                --answers-file ${ANSWERS_DIR}/chunk${IDX}.jsonl \
                --num-chunks $NUM_CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode ${CONV} \
                ${EXTRA_ARGS} &

            echo "${MASTER_PORT} $!" >> /tmp/running_processes.txt
        done
    else
        for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
            export NPU_VISIBLE_DEVICES=$(($IDX % 8))
            export ASCEND_RT_VISIBLE_DEVICES=$(($IDX % 8))
            export MASTER_PORT=$(get_available_port)

            # Additional args for SQA
            EXTRA_ARGS=""
            if [[ $EVAL_DATASET == "sqa" ]]; then
                EXTRA_ARGS="--single-pred-prompt"
            fi

            python3 -m ${EVAL_DENSE_SCRIPT} \
                --model-path ${CKPT} \
                --question-file ${QUESTION_FILE} \
                --image-folder ${IMAGE_FOLDER} \
                --answers-file ${ANSWERS_DIR}/chunk${IDX}.jsonl \
                --num-chunks $NUM_CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode ${CONV} \
                ${EXTRA_ARGS} &

            echo "${MASTER_PORT} $!" >> /tmp/running_processes.txt
        done
    fi

    wait

    cd "${ORIGINAL_DIR}"
    echo "Finished inference for ${CKPT} on ${EVAL_DATASET}!"
}

function calculate_scores() {
    local CKPT_NAME=$1
    local EVAL_DATASET=$2
    local BASE=$3
    local ORIGINAL_DIR=$(pwd)

    MOE_HOME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
    EVAL="/root/eval"
    CKPT="${BASE}/${CKPT_NAME}"

    cd ${MOE_HOME}

    case $EVAL_DATASET in
        "pope")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/pope/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl

            mkdir -p "${CKPT}/eval/pope"
            python3 moellava/eval/eval_pope_.py \
                --annotation-dir ${EVAL}/pope/coco \
                --question-file ${EVAL}/pope/llava_pope_test.jsonl \
                --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
                --output-dir "${CKPT}/eval/pope"
            ;;
        "mmbench")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/mmbench/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/mmbench/answers/${CKPT_NAME}.jsonl

            mkdir -p ${EVAL}/mmbench/answers_upload/${CKPT_NAME}
            python3 scripts/convert_mmbench_for_submission.py \
                --annotation-file ${EVAL}/mmbench/${SPLIT}.tsv \
                --result-dir ${EVAL}/mmbench/answers/${CKPT_NAME} \
                --upload-dir ${EVAL}/mmbench/answers_upload/${CKPT_NAME} \
                --experiment ${CKPT_NAME}
            ;;
        "mme")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/MME/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl

            cd ${EVAL}/MME
            python convert_answer_to_mme.py --experiment ${CKPT_NAME}

            cd eval_tool
            mkdir -p "${CKPT}/eval/MME"
            python calc.py --results_dir answers/${CKPT_NAME} --output_dir "${CKPT}/eval/MME"
            ;;
        "mmvet")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/mm-vet/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/mm-vet/answers/${CKPT_NAME}/${CKPT_NAME}.jsonl

            python3 scripts/convert_mmvet_for_eval.py \
                --src ${EVAL}/mm-vet/answers/${CKPT_NAME}/${CKPT_NAME}.jsonl \
                --dst ${EVAL}/mm-vet/results/${CKPT_NAME}.json
            cp ${EVAL}/mm-vet/results/${CKPT_NAME}.json /apdcephfs_nj7/share_1273717/yannhua/Home
            ;;
        "gqa")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}/ \
                --output-file ${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}/merge.jsonl

            mkdir -p ${EVAL}/gqa/data/testdev/${CKPT_NAME}
            mkdir -p ${EVAL}/gqa/data/llava_gqa_testdev_balanced/${CKPT_NAME}
            mkdir -p "${CKPT}/eval/gqa"
            python3 scripts/convert_gqa_for_eval.py \
                --src ${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}/merge.jsonl \
                --dst ${EVAL}/gqa/data/llava_gqa_testdev_balanced/${CKPT_NAME}/testdev_balanced_predictions.json

            python3 moellava/eval/eval_gqa_.py \
                --predictions ${EVAL}/gqa/data/llava_gqa_testdev_balanced/${CKPT_NAME}/testdev_balanced_predictions.json \
                --questions ${EVAL}/gqa/data/testdev_balanced_questions.json \
                --output-dir "${CKPT}/eval/gqa"
            ;;
        "sqa")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/scienceqa/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl

            python3 moellava/eval/eval_science_qa.py \
                --base-dir ${EVAL}/scienceqa \
                --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
                --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
                --output-result ${EVAL}/scienceqa/answers/${CKPT_NAME}_result.json
            ;;
        "textvqa")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/textvqa/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl

            mkdir -p "${CKPT}/eval/textvqa"
            python3 -m moellava.eval.eval_textvqa_ \
                --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
                --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl \
                --output-dir "${CKPT}/eval/textvqa"
            ;;
        "vizwiz")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/vizwiz/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl

            python3 scripts/convert_vizwiz_for_submission.py \
                --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
                --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
                --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json
            ;;
        *)
            echo "Unsupported dataset: ${EVAL_DATASET}"
            return 1
            ;;
    esac
    
    cd "${ORIGINAL_DIR}"
    echo "Finished evaluation for ${CKPT} on ${EVAL_DATASET}!"
}

function run_from_string() {
    local config_str="$1"
    local mode="$2"
    local use_moe="$3"  # New parameter for MoE flag

    local temp_configs=()
    
    while IFS= read -r line; do
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue
        temp_configs+=("$line")
    done < <(echo -e "$config_str")
    
    if [[ "$mode" == "full" ]]; then
        for config in "${temp_configs[@]}"; do
            IFS="|" read -r base model dataset <<< "${config}"
            echo "Starting inference for ${model} on ${dataset} from ${base}"
            run_inference "${base}" "${model}" "${dataset}" "${use_moe}" &  # Pass MoE flag
        done
        
        wait
        echo "All inferences completed"
    fi
    
    for config in "${temp_configs[@]}"; do
        IFS="|" read -r base model dataset <<< "${config}"
        calculate_scores "${model}" "${dataset}" "${base}"
    done

    echo "Running evaluation from config string in $mode mode"
    echo "Config string:"
    echo -e "$config_str"
}

function run_from_config() {
    local config_file="$1"
    local mode="$2"
    local use_moe="$3"  # New parameter for MoE flag
    local temp_configs=()
    
    while IFS="|" read -r base model dataset; do
        [[ $base =~ ^#.*$ ]] && continue
        [[ -z $base ]] && continue
        temp_configs+=("${base}|${model}|${dataset}")
    done < "${config_file}"
    
    if [[ "$mode" == "full" ]]; then
        for config in "${temp_configs[@]}"; do
            IFS="|" read -r base model dataset <<< "${config}"
            run_inference "${base}" "${model}" "${dataset}" "${use_moe}" &  # Pass MoE flag
            echo "Started inference for ${model} on ${dataset}"
        done
        
        wait
        echo "All inferences completed"
    fi
    
    for config in "${temp_configs[@]}"; do
        IFS="|" read -r base model dataset <<< "${config}"
        calculate_scores "${model}" "${dataset}" "${base}"
    done
}

function generate_eval_string() {
    local model_paths=($1)   # Space-separated list of full model paths
    local datasets=($2)      # Space-separated list of dataset names
    local config_string=""
    
    # Generate config string for each combination
    for model_path in "${model_paths[@]}"; do
        local base_dir=$(dirname "$model_path")
        local model_name=$(basename "$model_path")
        
        for dataset in "${datasets[@]}"; do
            if [ -n "$config_string" ]; then
                config_string+=$'\n'  # Use $'\n' for actual newline
            fi
            config_string+="${base_dir}|${model_name}|${dataset}"
        done
    done
    
    echo -e "$config_string"  # Use echo -e to interpret escape sequences
}

# Initialize variables
config_file=""
config_str=""
model_paths=()
datasets=()
mode="full"
use_moe="false"  # New variable for MoE flag

function print_usage() {
    echo "Usage:" >&2
    echo "  1. $0 --config <config_file> [--mode <mode>] [--moe]" >&2
    echo "  2. $0 --config-str <config_string> [--mode <mode>] [--moe]" >&2
    echo "  3. $0 --model_paths <path1> [path2 ...] --datasets <dataset1> [dataset2 ...] [--mode <mode>] [--moe]" >&2
    echo >&2
    echo "Options:" >&2
    echo "  --config      : Path to config file" >&2
    echo "  --config-str  : Configuration string" >&2
    echo "  --model_paths : One or more full paths to models" >&2
    echo "  --datasets    : One or more dataset names" >&2
    echo "  --mode        : Evaluation mode (full or calc_only, default: full)" >&2
    echo "  --moe         : Use deepspeed for MoE models" >&2
    exit 1
}


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            config_file="$2"
            shift 2
            ;;
        --config-str)
            config_str="$2"
            shift 2
            ;;
        --model_paths)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                model_paths+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --moe)
            use_moe="true"
            shift
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage
            ;;
    esac
done
# Validate mode
if [ "$mode" != "full" ] && [ "$mode" != "calc_only" ]; then
    echo "Error: Invalid mode '$mode'. Must be 'full' or 'calc_only'" >&2
    print_usage
fi

# Count how many input methods were specified
input_methods=0
[ -n "$config_file" ] && ((input_methods++))
[ -n "$config_str" ] && ((input_methods++))
[ ${#model_paths[@]} -gt 0 ] || [ ${#datasets[@]} -gt 0 ] && ((input_methods++))

# Validate inputs
if [ $input_methods -eq 0 ]; then
    echo "Error: Must specify one of: --config, --config-str, or --model_paths/--datasets" >&2
    print_usage
fi

if [ $input_methods -gt 1 ]; then
    echo "Error: Cannot specify multiple input methods simultaneously" >&2
    print_usage
fi


# Process according to input method
if [ -n "$config_file" ]; then
    run_from_config "$config_file" "$mode" "$use_moe"
elif [ -n "$config_str" ]; then
    run_from_string "$config_str" "$mode" "$use_moe"
else
    # Validate generate_eval_list parameters
    if [[ ${#model_paths[@]} -eq 0 ]] || [[ ${#datasets[@]} -eq 0 ]]; then
        echo "Error: When using generate mode, must specify --model_paths and --datasets" >&2
        print_usage
    fi
    
    config_str=$(generate_eval_string "${model_paths[*]}" "${datasets[*]}")
    run_from_string "$config_str" "$mode" "$use_moe"
fi
