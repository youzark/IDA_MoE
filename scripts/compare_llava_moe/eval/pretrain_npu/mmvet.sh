#!/bin/bash

export NPU_VISIBLE_DEVICES=8,9,10,11,12,13
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5
CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/root/eval"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m moellava.eval.model_vqa \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${EVAL}/mm-vet/images \
    --answers-file ${EVAL}/mm-vet/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --image_tower_cache_dir ${IMAGE_TOWER_FOLDER} \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/mm-vet/results

python3 scripts/convert_mmvet_for_eval.py \
    --src ${EVAL}/mm-vet/answers/${CKPT_NAME}.jsonl \
    --dst ${EVAL}/mm-vet/results/${CKPT_NAME}.json


python3 moellava/eval/eval_gpt_mmvet.py \
    --mmvet_path ${EVAL}/mm-vet \
    --ckpt_name ${CKPT_NAME} \
    --result_path ${EVAL}/mm-vet/results
