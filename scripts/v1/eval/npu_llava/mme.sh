#!/bin/bash

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune"
CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints/${CKPT_NAME}"
EVAL="/root/eval"
NUM_CHUNKS=8


mkdir -p ${EVAL}/MME/answers/${CKPT_NAME}
mkdir -p ${EVAL}/MME/results/${CKPT_NAME}
cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

for IDX in $(seq 0 $((NUM_CHUNKS-1))); 
do
    export NPU_VISIBLE_DEVICES=$(($IDX % 8))
    export ASCEND_RT_VISIBLE_DEVICES=$(($IDX % 8))

    python3 -m moellava.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${EVAL}/MME/llava_mme.jsonl \
        --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
        --answers-file ${EVAL}/MME/answers/${CKPT_NAME}/${CKPT_NAME}_chunk${IDX}.jsonl \
        --num-chunks $NUM_CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

echo "all finished!"

python moellava/eval/merge_mmvet_chunk.py \
    --input-folder ${EVAL}/MME/answers/${CKPT_NAME}/ \
    --output-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl

cd $EVAL/MME

python convert_answer_to_mme.py --experiment ${CKPT_NAME}

cd eval_tool

python calculation.py --results_dir answers/${CKPT_NAME}

