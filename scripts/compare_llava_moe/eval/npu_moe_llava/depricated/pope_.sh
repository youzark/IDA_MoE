#!/bin/bash

function run_inference() {
    local BASE=$1
    local CKPT_NAME=$2
    local BASE_MASTER_PORT=$((29500 + RANDOM % 1000))

    CONV="qwen"
    MOE_HOME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
    CKPT="${BASE}/${CKPT_NAME}"
    EVAL="/root/eval"
    NUM_CHUNKS=8

    mkdir -p ${EVAL}/pope/answers/${CKPT_NAME}
    cd ${MOE_HOME}

    if [[ $CKPT_NAME == *"moe"* ]]; then
        # MOE version using deepspeed
        for IDX in $(seq 0 $((NUM_CHUNKS-1))); 
        do
            MASTER_PORT=$((BASE_MASTER_PORT + IDX))
            INCLUDE_DEVICES="localhost:${IDX}"

            deepspeed \
                --master_port $MASTER_PORT \
                --include $INCLUDE_DEVICES \
                moellava/eval/model_vqa_loader.py \
                --model-path ${CKPT} \
                --question-file ${EVAL}/pope/llava_pope_test.jsonl \
                --image-folder ${EVAL}/pope/val2014 \
                --answers-file ${EVAL}/pope/answers/${CKPT_NAME}/chunk${IDX}.jsonl \
                --num-chunks $NUM_CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode ${CONV} &
        done
    else
        # Non-MOE version
        for IDX in $(seq 0 $((NUM_CHUNKS-1))); 
        do
            export NPU_VISIBLE_DEVICES=$(($IDX % 8))
            export ASCEND_RT_VISIBLE_DEVICES=$(($IDX % 8))
            export MASTER_PORT=$BASE_MASTER_PORT

            python3 -m moellava.eval.model_vqa_loader \
                --model-path ${CKPT} \
                --question-file ${EVAL}/pope/llava_pope_test.jsonl \
                --image-folder ${EVAL}/pope/val2014 \
                --answers-file ${EVAL}/pope/answers/${CKPT_NAME}/chunk${IDX}.jsonl \
                --num-chunks $NUM_CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode ${CONV} &
        done
    fi

    wait
    echo "Finished inference for ${CKPT_NAME}!"
}

function calculate_scores() {
    local CKPT_NAME=$1
    MOE_HOME="/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
    EVAL="/root/eval"

    cd ${MOE_HOME}
    python moellava/eval/merge_mmvet_chunk.py \
        --input-folder ${EVAL}/pope/answers/${CKPT_NAME}/ \
        --output-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl

    python3 moellava/eval/eval_pope.py \
        --annotation-dir ${EVAL}/pope/coco \
        --question-file ${EVAL}/pope/llava_pope_test.jsonl \
        --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl
    
    echo "Finished evaluation for ${CKPT_NAME}!"
}


# run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe" "llavaqwen-1.8b-finetune-moe-sharpen_v4" &
# run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe" "llavaqwen-1.8b-finetune-moe_v4" &
run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"  "llavaqwen-1.8b-finetune-moe-normal_0.01"&
run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"  "llavaqwen-1.8b-finetune-moe-sharpen_0.1"&
run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"  "llavaqwen-1.8b-finetune-moe-sharpen_0.8"&
run_inference "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe"  "llavaqwen-1.8b-finetune-moe-sharpen_1"&
wait

# Then calculate scores sequentially
# calculate_scores "llavaqwen-1.8b-finetune-moe-sharpen_v4"
# calculate_scores "llavaqwen-1.8b-finetune-moe_v4"
calculate_scores "llavaqwen-1.8b-finetune-moe-normal_0.01"
calculate_scores "llavaqwen-1.8b-finetune-moe-sharpen_0.1"
calculate_scores "llavaqwen-1.8b-finetune-moe-sharpen_0.8"
calculate_scores "llavaqwen-1.8b-finetune-moe-sharpen_1"
