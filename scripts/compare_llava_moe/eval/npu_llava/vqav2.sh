#!/bin/bash

export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,7
npu_list="${ASCEND_RT_VISIBLE_DEVICES:-0}"
IFS=',' read -ra NPULIST <<< "$npu_list"

CHUNKS=${#NPULIST[@]}
echo $CHUNKS

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune"
CKPT="checkpoints/${CKPT_NAME}"
SPLIT="llava_vqav2_mscoco_test-dev2015"
EVAL="/root/eval"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"


cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launch Process with IDX: $IDX"
    ASCEND_RT_VISIBLE_DEVICES=${NPULIST[$IDX]} python3 -m moellava.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${EVAL}/vqav2/$SPLIT.jsonl \
        --image-folder ${EVAL}/vqav2/test2015 \
        --answers-file ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --image_tower_cache_dir ${IMAGE_TOWER_FOLDER} \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt ${CKPT_NAME} --dir ${EVAL}/vqav2
