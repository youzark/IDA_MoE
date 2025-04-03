#!/bin/bash
function get_available_port() {
    local BASE_PORT=29500
    local MAX_PORT=30500
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
            QUESTION_FILE="${EVAL}/pope/llava_pope_test.jsonl"
            IMAGE_FOLDER="${EVAL}/pope/val2014"
            ANSWERS_DIR="${EVAL}/pope/answers/${CKPT_NAME}"
            ;;
        "mme")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            QUESTION_FILE="${EVAL}/MME/llava_mme.jsonl"
            IMAGE_FOLDER="${EVAL}/MME/MME_Benchmark_release_version"
            ANSWERS_DIR="${EVAL}/MME/answers/${CKPT_NAME}"
            mkdir -p ${EVAL}/MME/results/${CKPT_NAME}
            ;;
        "mmvet")
            EVAL_SCRIPT="moellava/eval/model_vqa.py"
            QUESTION_FILE="${EVAL}/mm-vet/llava-mm-vet.jsonl"
            IMAGE_FOLDER="${EVAL}/mm-vet/images"
            ANSWERS_DIR="${EVAL}/mm-vet/answers/${CKPT_NAME}"
            ;;
        "gqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            QUESTION_FILE="${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl"
            IMAGE_FOLDER="${EVAL}/gqa/data/images"
            ANSWERS_DIR="${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}"
            ;;
        "sqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_science.py"
            QUESTION_FILE="${EVAL}/scienceqa/llava_test_CQM-A.json"
            IMAGE_FOLDER="${EVAL}/scienceqa/images/test"
            ANSWERS_DIR="${EVAL}/scienceqa/answers"
            ;;
        "textvqa")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            QUESTION_FILE="${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl"
            IMAGE_FOLDER="${EVAL}/textvqa/train_images"
            ANSWERS_DIR="${EVAL}/textvqa/answers"
            ;;
        "vizwiz")
            EVAL_SCRIPT="moellava/eval/model_vqa_loader.py"
            QUESTION_FILE="${EVAL}/vizwiz/llava_test.jsonl"
            IMAGE_FOLDER="${EVAL}/vizwiz/test"
            ANSWERS_DIR="${EVAL}/vizwiz/answers"
            ;;
        *)
            echo "Unsupported dataset: ${EVAL_DATASET}"
            return 1
            ;;
    esac

    mkdir -p ${ANSWERS_DIR}
    cd ${MOE_HOME}

    # All datasets use parallel processing
    if [[ $CKPT_NAME == *"moe"* ]]; then
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

            python3 -m ${EVAL_SCRIPT} \
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
    echo "Finished inference for ${CKPT_NAME} on ${EVAL_DATASET}!"
}

function calculate_scores() {
    local CKPT_NAME=$1
    local EVAL_DATASET=$2
    local ORIGINAL_DIR=$(pwd)

    MOE_HOME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
    EVAL="/root/eval"

    cd ${MOE_HOME}

    case $EVAL_DATASET in
        "pope")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/pope/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl

            python3 moellava/eval/eval_pope.py \
                --annotation-dir ${EVAL}/pope/coco \
                --question-file ${EVAL}/pope/llava_pope_test.jsonl \
                --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl
            ;;
        "mme")
            python moellava/eval/merge_mmvet_chunk.py \
                --input-folder ${EVAL}/MME/answers/${CKPT_NAME}/ \
                --output-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl

            cd ${EVAL}/MME
            python convert_answer_to_mme.py --experiment ${CKPT_NAME}

            cd eval_tool
            python calculation.py --results_dir answers/${CKPT_NAME}
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
            python3 scripts/convert_gqa_for_eval.py \
                --src ${EVAL}/gqa/answers/llava_gqa_testdev_balanced/${CKPT_NAME}/merge.jsonl \
                --dst ${EVAL}/gqa/data/llava_gqa_testdev_balanced/${CKPT_NAME}/testdev_balanced_predictions.json

            cd ${EVAL}/gqa/data
            python3 eval/eval_gqa.py \
                --tier llava_gqa_testdev_balanced/${CKPT_NAME}/testdev_balanced \
                --questions ${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json
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

            python3 -m moellava.eval.eval_textvqa \
                --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
                --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl
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
    echo "Finished evaluation for ${CKPT_NAME} on ${EVAL_DATASET}!"
}

function run_from_config() {
    local config_file=$1
    local mode=$2  # "full" or "calc_only"
    local temp_configs=()
    
    # Read and store valid configurations
    while IFS="|" read -r base model dataset; do
        [[ $base =~ ^#.*$ ]] && continue  # Skip commented lines
        [[ -z $base ]] && continue        # Skip empty lines
        temp_configs+=("${base}|${model}|${dataset}")
    done < "${config_file}"
    
    if [[ "$mode" == "full" ]]; then
        # Run all inferences in parallel from stored configs
        for config in "${temp_configs[@]}"; do
            IFS="|" read -r base model dataset <<< "${config}"
            run_inference "${base}" "${model}" "${dataset}" &
            echo "Started inference for ${model} on ${dataset}"
        done
        
        # Wait for all inferences to complete
        wait
        echo "All inferences completed"
    fi
    
    # Run calculations sequentially from stored configs
    for config in "${temp_configs[@]}"; do
        IFS="|" read -r base model dataset <<< "${config}"
        calculate_scores "${model}" "${dataset}"
    done
}

function run_from_string() {
    local config_string=$1
    local mode=${2:-full}  # Default to "full" if not specified
    local temp_configs=()
    
    # Read and store valid configurations from the string
    while IFS="|" read -r base model dataset; do
        [[ $base =~ ^#.*$ ]] && continue  # Skip commented lines
        [[ -z $base ]] && continue        # Skip empty lines
        temp_configs+=("${base}|${model}|${dataset}")
    done <<< "$config_string"
    
    if [[ "$mode" == "full" ]]; then
        # Run all inferences in parallel from stored configs
        for config in "${temp_configs[@]}"; do
            IFS="|" read -r base model dataset <<< "${config}"
            run_inference "${base}" "${model}" "${dataset}" &
            echo "Started inference for ${model} on ${dataset}"
        done
        
        # Wait for all inferences to complete
        wait
        echo "All inferences completed"
    fi
    
    # Run calculations sequentially from stored configs
    for config in "${temp_configs[@]}"; do
        IFS="|" read -r base model dataset <<< "${config}"
        calculate_scores "${model}" "${dataset}"
    done
}

# Script usage check
function print_usage() {
    echo "Usage: $0 <config_file> [mode]"
    echo "Modes:"
    echo "  full      - Run both inference and calculations (default)"
    echo "  calc_only - Run only calculations"
    exit 1
}

# Main script
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    print_usage
fi

config_file=$1
mode=${2:-full}  # Default to "full" if not specified

case $mode in
    "full"|"calc_only")
        run_from_config "$config_file" "$mode"
        ;;
    *)
        echo "Error: Invalid mode '$mode'"
        print_usage
        ;;
esac

