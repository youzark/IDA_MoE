EXP_BASE_DIR="/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data"

echo "==== Train According to MoELLaVA's Setting"
echo "BaseModel:\t stablelm1.6"
echo "Dataset:\t LLaVA DataSet"

cd "/apdcephfs_nj7/share_1273717/yannhua/Home/moe"

function pretrain_sig() {
    bash ./scripts/compare_llava_moe/pretrain.sh \
        --data-paths /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_.json \
        --image-folder /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b \
        --llm-folder "stabilityai/stablelm-2-1_6b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --output-ckpt-name "llavastablelm-siglip-1.6b-pretrain" | tee ./exps/stablelm/MoELLaVA-Data/ptSig.txt
}

function finetune_sig() {
    bash ./scripts/compare_llava_moe/finetune.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --llm-folder "stabilityai/stablelm-2-1_6b" \
        --outdir $EXP_BASE_DIR \
        --version stablelm \
        --start-ckpt-name "llavastablelm-siglip-1.6b-pretrain" \
        --output-ckpt-name "llavastablelm-siglip-1.6b-finetune" | tee ./exps/stablelm/MoELLaVA-Data/ftSig.txt
}

function gmm_c16_d32_sig_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-siglip-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-siglip-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmSigC16D32T1_Base.txt
}

function gmm_c16_d32_sig() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --image-tower "google/siglip-so400m-patch14-384" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-siglip-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-Base" \
        --output-ckpt-name "llavastablelm-siglip-1.6b-finetune-moe-gmm-1.0-center-16-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmSigC16D32.txt
}

function pretrain() {
    bash ./scripts/compare_llava_moe/pretrain.sh \
        --data-paths /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_.json \
        --image-folder /root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b \
        --llm-folder "stabilityai/stablelm-2-1_6b" \
        --outdir $EXP_BASE_DIR \
        --output-ckpt-name "llavastablelm-1.6b-pretrain" | tee ./exps/stablelm/MoELLaVA-Data/pt.txt
}

function finetune() {
    bash ./scripts/compare_llava_moe/finetune.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --llm-folder "stabilityai/stablelm-2-1_6b" \
        --outdir $EXP_BASE_DIR \
        --version stablelm \
        --start-ckpt-name "llavastablelm-1.6b-pretrain" \
        --output-ckpt-name "llavastablelm-1.6b-finetune" | tee ./exps/stablelm/MoELLaVA-Data/ft.txt
}

function dense() {
    bash ./scripts/compare_llava_moe/finetune.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --llm-folder "stabilityai/stablelm-2-1_6b" \
        --outdir $EXP_BASE_DIR \
        --version stablelm \
        --start-ckpt-name "llavastablelm-1.6b-pretrain" \
        --output-ckpt-name "llavastablelm-1.6b-dense" | tee ./exps/stablelm/MoELLaVA-Data/dense.txt
}

function lb001() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-loadbalancing-0.01" \
        --train-modules gate_proj up_proj down_proj wg \
        --version stablelm \
        --l-aux-type "load_balancing" \
        --batch-size 2 \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.008 | tee ./exps/stablelm/MoELLaVA-Data/lb001.txt
}

function router_z() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-router-z-0.01" \
        --train-modules gate_proj up_proj down_proj wg projection \
        --version stablelm \
        --l-aux-type "router_z" \
        --routing-dim 16 \
        --batch-size 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 | tee ./exps/stablelm/MoELLaVA-Data/router_z.txt
}

function xMoE001() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-xMoE-0.01" \
        --train-modules gate_proj up_proj down_proj wg projection \
        --version stablelm \
        --l-aux-type "xMoE" \
        --routing-dim 16 \
        --batch-size 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 | tee ./exps/stablelm/MoELLaVA-Data/xMoE.txt
}

function DeepSeekMoE0001() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-DeepSeekMoE-0.001" \
        --train-modules gate_proj up_proj down_proj wg coefficient\
        --version stablelm \
        --l-aux-type "load_balancing" \
        --batch-size 2 \
        --use-residual True \
        --top-k-experts 1 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.001 | tee ./exps/stablelm/MoELLaVA-Data/DeepSeekMoE.txt
}


function sp020() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-sharpen-0.2" \
        --train-modules gate_proj up_proj down_proj wg \
        --version stablelm \
        --l-aux-type "sharpen" \
        --router-aux-loss-coef 0.2 | tee ./exps/stablelm/MoELLaVA-Data/sp020.txt
}

function gmm_c16_d32_t1_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1_Base.txt
}

function gmm_c16_d32_t1() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-test" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1.txt
}
        # --localhost 0 \

function gmm_c32_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-32-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 32 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC32D32_Base.txt
}

function gmm_c32_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-32-dim-32-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-32-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 32 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC32D32.txt
}

function gmm_c4_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-4-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 4 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC4D32_Base.txt
}

function gmm_c4_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-4-dim-32-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-4-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 4 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC4D32.txt
}


function gmm_c1_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-1-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 1 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC1D32_Base.txt
}

function gmm_c1_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-1-dim-32-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-1-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 1 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC1D32.txt
}

function gmm_c8_d32_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-8-dim-32-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 8 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC8D32_Base.txt
}

function gmm_c8_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-8-dim-32-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-8-dim-32" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 8 \
        --batch-size 4 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC8D32.txt
}

function gmm_c16_d32() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-reproduce" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1.txt
}


function gmm_c16_d16_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-16-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 16| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D16T1_Base.txt
}

function gmm_c16_d16() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-16-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-16" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 16| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D16T1.txt
}


function gmm_c16_d8_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-8-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 8| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D8T1_Base.txt
}

function gmm_c16_d8() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-8-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-8" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 8| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D8T1.txt
}

function gmm_c16_d4_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-4-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 4| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D4_Base.txt
}

        # --localhost 0 \
function gmm_c16_d4() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-4-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-4" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 4| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D4.txt
}


function gmm_c16_d64_Base() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-64-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 64| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D64_Base.txt
}

function gmm_c16_d64() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-64-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-64" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 2 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 64| tee ./exps/stablelm/MoELLaVA-Data/gmmC16D64.txt
}

        # --localhost 0 \
function gmm_c16_d32_Base_No_reactivation() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-noReact-Base" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32Base.txt
}

function gmm_c16_d32_reproduce() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-noReact-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-noReact" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32.txt
}

function gmm_c16_d32_routing_noReact() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-NoReact-Routing" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --group-reactivation False \
        --routing_file_path "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/Routing/No-React.jsonl"\
        --if_training_for_routing_analysis True \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32.txt
}

function gmm_c16_d32_stable_noReact() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-NoReact-Routing" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-NoReact-Routing-No_train" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --group-reactivation False \
        --routing_file_path "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/Routing/No-React-After-Train.jsonl"\
        --if_training_for_routing_analysis False \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32_No_training.txt
}


function lb000_routing() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-no-loadbalancing" \
        --train-modules gate_proj up_proj down_proj wg \
        --version stablelm \
        --batch-size 1 \
        --gradient-accumulation-steps 4 \
        --l-aux-type "load_balancing" \
        --routing_file_path "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/Routing/MoE_no_lb_routing.jsonl"\
        --if_training_for_routing_analysis True \
        --router-aux-loss-coef 0 | tee ./exps/stablelm/MoELLaVA-Data/lb0.txt
}

function lb000_routing_no_train() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-no-loadbalancing" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-no-loadbalancing-No_train" \
        --train-modules gate_proj up_proj down_proj wg \
        --batch-size 1 \
        --gradient-accumulation-steps 4 \
        --version stablelm \
        --l-aux-type "load_balancing" \
        --routing_file_path "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/Routing/MoE_no_lb_routing-After-Train.jsonl"\
        --if_training_for_routing_analysis False \
        --router-aux-loss-coef 0 | tee ./exps/stablelm/MoELLaVA-Data/lb0_NoTrain.txt
}

function AuxFree0001() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-AuxFree-0.001" \
        --train-modules gate_proj up_proj down_proj wg expert_BIAS \
        --version stablelm \
        --l-aux-type "aux_free" \
        --batch-size 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 2 \
        --router-aux-loss-coef 0.001 | tee ./exps/stablelm/MoELLaVA-Data/AuxFree0001.txt
}

function gmm_c16_d32_t1_Base_0.2Data() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lrv_tune_331k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/lvis_tune_220k_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/svit_tune_157k.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --epoch 0.2 \
        --start-ckpt-name "llavastablelm-1.6b-finetune" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base-0.2Data" \
        --train-modules means vars mix_logits projection \
        --components-per-expert 16 \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing False \
        --gradient-accumulation-steps 4 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1_Base.txt
}

function gmm_c16_d32_no_React() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-noReact" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --group-reactivation False \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1.txt
}

function gmm_c16_d32_direct() {
    bash ./scripts/compare_llava_moe/finetune_moe.sh \
        --data-paths "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/llava_image_tune_.json" \
            "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/nlp_tune.json" \
        --image-folder "/root/data/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b" \
        --outdir $EXP_BASE_DIR \
        --start-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-Base-0.2Data" \
        --output-ckpt-name "llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-Direct" \
        --train-modules gate_proj up_proj down_proj means vars mix_logits projection \
        --components-per-expert 16 \
        --group-reactivation False \
        --batch-size 1 \
        --version stablelm \
        --l-aux-type "gaussian" \
        --router-aux-loss-coef 1 \
        --top-k-experts 2 \
        --gradient-checkpointing True \
        --gradient-accumulation-steps 4 \
        --router-aux-loss-coef 0.01 \
        --routing-dim 32  | tee ./exps/stablelm/MoELLaVA-Data/gmmC16D32T1.txt
}


# pretrain
# finetune
# lb001
# gmm_c16_d32_t1_Base
# gmm_c16_d32_t1
# gmm_c32_d32_Base
# gmm_c32_d32
# gmm_c4_d32_Base
# gmm_c4_d32
# gmm_c16_d32
# gmm_c1_d32_Base
# gmm_c16_d32
# gmm_c16_d16_Base
# gmm_c16_d16
# gmm_c16_d8_Base
# gmm_c16_d8
# gmm_c16_d32_Base_No_reactivation
# gmm_c16_d32_reproduce
# gmm_c16_d4_Base
# gmm_c16_d4
# dense
# gmm_c8_d32_Base
# gmm_c8_d32
# gmm_c16_d32_routing_noReact
# gmm_c16_d32_stable_noReact
# xMoE001
# lb000_routing
# lb000_routing_no_train
# DeepSeekMoE0001
# gmm_c16_d64_Base
# gmm_c16_d64
# AuxFree0001
# gmm_c16_d32_t1_Base_0.2Data
# gmm_c16_d32_direct
# gmm_c16_d32_no_React
# pretrain_sig
# finetune_sig
# gmm_c16_d32_sig_Base
# gmm_c16_d32_sig
# lb001
router_z