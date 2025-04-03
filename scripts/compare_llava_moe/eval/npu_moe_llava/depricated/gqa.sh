#!/bin/bash

CHUNKS=8

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune-moe"
CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe/${CKPT_NAME}"
EVAL="/root/eval"
GQADIR="${EVAL}/gqa/data"
cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

for IDX in $(seq 0 $((NUM_CHUNKS-1))); 
do
    export NPU_VISIBLE_DEVICES=$(($IDX % 8))
    export ASCEND_RT_VISIBLE_DEVICES=$(($IDX % 8))
    MASTER_PORT=$((29500 + CHUNK_IDX))
    INCLUDE_DEVICES="localhost:${CHUNK_IDX}"

    deepspeed \
        --master_port $MASTER_PORT \
        --include $INCLUDE_DEVICES \
        moellava/eval/model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${EVAL}/MME/llava_mme.jsonl \
        --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
        --answers-file ${EVAL}/MME/answers/${CKPT_NAME}/${CKPT_NAME}-${IDX}.jsonl \
        --num-chunks $NUM_CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); 
do
    export NPU_VISIBLE_DEVICES=$(($IDX % 8))
    export ASCEND_RT_VISIBLE_DEVICES=$(($IDX % 8))
    MASTER_PORT=$((29500 + CHUNK_IDX))
    INCLUDE_DEVICES="localhost:${CHUNK_IDX}"

    deepspeed \
        --master_port $MASTER_PORT \
        --include $INCLUDE_DEVICES \
        moellava/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL}/gqa/data/images \
        --answers-file ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $GQADIR/$SPLIT/${CKPT_NAME}
python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${CKPT_NAME}/testdev_balanced_predictions.json

cd $GQADIR
python3 eval/eval_gqa.py --tier $SPLIT/${CKPT_NAME}/testdev_balanced \
                         --questions ${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json
