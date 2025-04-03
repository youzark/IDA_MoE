EXP_BASE_DIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2"

echo "==== Train According to MoELLaVA's Setting"
echo "BaseModel:\t Qwen2-1.5B"
echo "Dataset:\t LLaVA DataSet"

cd "/apdcephfs_nj7/share_1273717/yannhua/Home/moe"

function pretrain() {
    bash ./scripts/compare_llava_moe/pretrain.sh \
        --data-paths /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_.json \
        --image-folder /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b \
        --llm-folder "Qwen/Qwen2-1.5B" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --output-ckpt-name "llavaqwen2-1.5b-pretrain" | tee ./exps/qwen2/pt.txt
}

function finetune() {
    bash ./scripts/compare_llava_moe/finetune.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --llm-folder "Qwen/Qwen2-1.5B" \
        --outdir $EXP_BASE_DIR \
        --version qwen \
        --start-ckpt-name "llavaqwen2-1.5b-pretrain" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune" | tee ./exps/qwen2/ft.txt
}


function gmm_c16_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version qwen \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/qwen2/gmmC16D32_Base.txt
}

function gmm_c16_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32-Base" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version qwen \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/qwen2/gmmC16D32.txt
}


function gmm_c16_d32_bunny() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths \
            "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/bunny_695k.json" \
            "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/bunny_allava_1.3m.json" \
            "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/bunny_llava_1.4m.json" \
            "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/bunny_llava_allava_2m.json" \
            "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/llava_v1_5_mix665k.json" \
        --image-folder "/root/.cache/huggingface/hub/datasets--BoyaWu10--Bunny-v1_1-data/snapshots/994013f36c94484c87a764916ee02ad377d3e9b0/finetune/images" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32-Base" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32-Bunny" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version qwen \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/qwen2/gmmC16D32.txt
}

function gmm_c16_e16_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-expert-16-center-16-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --num-experts 16 \
        --components-per-expert 16 \
        --batch-size 2 \
        --version qwen \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/qwen2/gmmE16C16D32_Base.txt
}

function gmm_c16_e16_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-expert-16-center-16-dim-32-Base" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-expert-16-center-16-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version qwen \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/qwen2/gmmE16C16D32.txt
}

function gmm_c16_d64_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-64-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version qwen \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --routing-dim 64  | tee ./exps/qwen2/gmmC16D64_Base.txt
}

function gmm_c16_d64() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-64-Base" \
        --output-ckpt-name "llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-64" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version qwen \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 64  | tee ./exps/qwen2/gmmC16D64.txt
}

# pretrain
# finetune
# gmm_c16_d32_Base
# gmm_c16_d32
# gmm_c16_e16_d32_Base
# gmm_c16_e16_d32
# gmm_c16_d32_bunny
gmm_c16_d64_Base
gmm_c16_d64