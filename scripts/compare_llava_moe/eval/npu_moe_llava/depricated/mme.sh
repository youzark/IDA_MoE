##!/bin/bash

#CONV="qwen"
#CKPT_NAME="llavaqwen-1.8b-finetune-moe-sharpen"
#CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe/${CKPT_NAME}"
#EVAL="/root/eval"
#NUM_CHUNKS=8


#mkdir -p ${EVAL}/MME/answers/${CKPT_NAME}
#mkdir -p ${EVAL}/MME/results/${CKPT_NAME}
#cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

#for IDX in $(seq 0 $((NUM_CHUNKS-1))); 
#do
#    MASTER_PORT=$((29510 + IDX))
#    INCLUDE_DEVICES="localhost:${IDX}"

#    deepspeed \
#        --master_port $MASTER_PORT \
#        --include $INCLUDE_DEVICES \
#        moellava/eval/model_vqa_loader.py \
#        --model-path ${CKPT} \
#        --question-file ${EVAL}/MME/llava_mme.jsonl \
#        --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
#        --answers-file ${EVAL}/MME/answers/${CKPT_NAME}/${CKPT_NAME}_chunk${IDX}.jsonl \
#        --num-chunks $NUM_CHUNKS \
#        --chunk-idx $IDX \
#        --temperature 0 \
#        --conv-mode ${CONV} &
#done

#wait

#echo "all finished!"

#python moellava/eval/merge_mmvet_chunk.py \
#    --input-folder ${EVAL}/MME/answers/${CKPT_NAME}/ \
#    --output-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl

#cd $EVAL/MME

#python convert_answer_to_mme.py --experiment ${CKPT_NAME}

#cd eval_tool

#python calculation.py --results_dir answers/${CKPT_NAME}

#!/bin/bash

function run_mme_eval() {
    local CKPT_NAME=$1
    local BASE_MASTER_PORT=$((29500 + RANDOM % 1000))  # Random base port between 29500-30500

    CONV="qwen"
    CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/compare_llava_moe/${CKPT_NAME}"
    EVAL="/root/eval"
    NUM_CHUNKS=8

    mkdir -p ${EVAL}/MME/answers/${CKPT_NAME}
    mkdir -p ${EVAL}/MME/results/${CKPT_NAME}
    cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

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
                --question-file ${EVAL}/MME/llava_mme.jsonl \
                --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
                --answers-file ${EVAL}/MME/answers/${CKPT_NAME}/${CKPT_NAME}_chunk${IDX}.jsonl \
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
                --question-file ${EVAL}/MME/llava_mme.jsonl \
                --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
                --answers-file ${EVAL}/MME/answers/${CKPT_NAME}/${CKPT_NAME}_chunk${IDX}.jsonl \
                --num-chunks $NUM_CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode ${CONV} &
        done
    fi

    wait

    echo "Finished evaluation for ${CKPT_NAME}!"

    python moellava/eval/merge_mmvet_chunk.py \
        --input-folder ${EVAL}/MME/answers/${CKPT_NAME}/ \
        --output-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl

    cd $EVAL/MME

    python convert_answer_to_mme.py --experiment ${CKPT_NAME}

    cd eval_tool

    python calculation.py --results_dir answers/${CKPT_NAME}
}

# Example usage for multiple checkpoints in parallel:
# run_mme_eval "checkpoint1-moe" &
# run_mme_eval "checkpoint2" &
# run_mme_eval "checkpoint3-moe" &
# wait
