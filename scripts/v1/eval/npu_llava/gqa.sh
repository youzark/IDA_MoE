#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
npu_list="${ASCEND_RT_VISIBLE_DEVICES:-0}"
IFS=',' read -ra NPULIST <<< "$npu_list"

CHUNKS=${#NPULIST[@]}

CONV="qwen"
CKPT_NAME="llavaqwen2.5-7b-finetune"
CKPT="checkpoints/${CKPT_NAME}"
SPLIT="llava_gqa_testdev_balanced"
EVAL="/root/eval"
GQADIR="${EVAL}/gqa/data"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
for IDX in $(seq 0 $((CHUNKS-1))); do
    ASCEND_RT_VISIBLE_DEVICES=${NPULIST[$IDX]} python3 -m moellava.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL}/gqa/data/images \
        --answers-file ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --image_tower_cache_dir ${IMAGE_TOWER_FOLDER} \
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

python3 moellava/eval/eval_gqa.py --tier $GQADIR/$SPLIT/${CKPT_NAME}/testdev_balanced \
                         --questions ${EVAL}/gqa/data/testdev_balanced_questions.json
