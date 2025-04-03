#!/bin/bash
CKPT_NAME="llava-v1.5-7b-finetune"
IMAGE_FOLDER="/root/data/LLaVA-Finetune"
JSON_FOLDER="/root/data/LLaVA-Finetune"
LLM_FOLDER="/root/model/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
PROJ_CKPT="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints"
cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    --include localhost:0,1,2,3,4,5,6,7 \
    moellava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_FOLDER}\
    --version v1 \
    --data_path $JSON_FOLDER/llava_v1_5_mix665k.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_tower_cache_dir ${IMAGE_TOWER_FOLDER} \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter $PROJ_CKPT \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTDIR/$CKPT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

    # --model_name_or_path Qwen/Qwen-1_8B \
    # --tf32 True \
    # --data_path ${JSON_FOLDER}/la_tune_256k.json \
    #             ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json \
    #             ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json \
