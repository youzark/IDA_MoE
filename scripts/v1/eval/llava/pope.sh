#!/bin/bash

CONV="stablelm"
CKPT_NAME="stablelm_test_pope"
CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-v2"
EVAL="/root/eval"
python3 -m moellava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL}/pope/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl
