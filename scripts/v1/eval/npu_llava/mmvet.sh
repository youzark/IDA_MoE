#!/bin/bash

CONV="qwen"
CKPT_NAME="llava-v1.5-7b-finetune"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/root/eval"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"


cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
NUM_CHUNKS=8

for CHUNK_IDX in $(seq 0 $(($NUM_CHUNKS - 1)))
do
    export NPU_VISIBLE_DEVICES=$(($CHUNK_IDX % 8))
    export ASCEND_RT_VISIBLE_DEVICES=$(($CHUNK_IDX % 8))

    mkdir -p ${EVAL}/mm-vet/answers/${CKPT_NAME}
    python3 -m moellava.eval.model_vqa \
        --model-path ${CKPT} \
        --question-file ${EVAL}/mm-vet/llava-mm-vet.jsonl \
        --image-folder ${EVAL}/mm-vet/images \
        --answers-file ${EVAL}/mm-vet/answers/${CKPT_NAME}/chunk${CHUNK_IDX}.jsonl \
        --temperature 0 \
        --conv-mode ${CONV} \
        --num-chunks ${NUM_CHUNKS} \
        --chunk-idx ${CHUNK_IDX} &
done
wait

python moellava/eval/merge_mmvet_chunk.py \
    --input-folder ${EVAL}/mm-vet/answers/${CKPT_NAME}/ \
    --output-file ${EVAL}/mm-vet/answers/${CKPT_NAME}/${CKPT_NAME}.jsonl

mkdir -p ${EVAL}/mm-vet/results

python3 scripts/convert_mmvet_for_eval.py \
    --src ${EVAL}/mm-vet/answers/${CKPT_NAME}/${CKPT_NAME}.jsonl \
    --dst ${EVAL}/mm-vet/results/${CKPT_NAME}.json
