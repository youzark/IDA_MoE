#!/bin/bash

IMAGE_FOLDER="/root/data/LLaVA-Pretrain/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/images"
JSON_FOLDER="/root/data/LLaVA-Pretrain/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727"
LLM_FOLDER="/root/model/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
IMAGE_TOWER_FOLDER="/root/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
OUTDIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/checkpoints"


cd /apdcephfs_nj7/share_1273717/yannhua/Home/moe
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_port 29500 \
    --include localhost:0,1,2,3,4,5,6,7 \
    moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM_FOLDER}\
    --version plain \
    --data_path ${JSON_FOLDER}/blip_laion_cc_sbu_558k.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_tower_cache_dir ${IMAGE_TOWER_FOLDER} \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTDIR/llava-v1.5-7b-pretrain\
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

    # --tf32 True \
