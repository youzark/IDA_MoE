MODEL_PATHS=(
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-loadbalancing-0.01"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-v2"
    "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-32-dim-32"
    "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-4-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-top1-reproduce"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-1-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-16"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-8"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-noReact"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-4"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-64"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-dense"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-8-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-xMoE-0.01"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-DeepSeekMoE-0.001"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-AuxFree-0.001"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-1.6b-finetune-moe-gmm-1.0-center-16-dim-32-Norm"
    # "/root/.cache/huggingface/hub/models--LanguageBind--MoE-LLaVA-StableLM-1.6B-4e/snapshots/e48f557cd69ca1a89419e54b00f9eb5d08cb2130"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/llavastablelm-siglip-1.6b-finetune-moe-gmm-1.0-center-16-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/"
)
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/stablelm/MoELLaVA-Data/"

# List of datasets
# DATASETS=("mmvet" "mme" "vizwiz" "pope" "sqa" "gqa" "textvqa" "vqav2" "mmbench")
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa" )
DATASETS=("mme" "pope")
# DATASETS=("mmbench")
# DATASETS=("mmbench-cn")
# DATASETS=("llavabench")
# DATASETS=("mme")
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa")
# DATASETS=("pope")
# DATASETS=("vizwiz")
# DATASETS=("vqav2")

cd "/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
# cd "/Users/yannhua/mnt/DevCloud/moe"
# Loop through each dataset

for dataset in "${DATASETS[@]}"; do
    # Concatenate all model paths
    model_paths_arg=""
    for model_path in "${MODEL_PATHS[@]}"; do
        model_paths_arg+=" $model_path"
    done

    # Run the eval script with concatenated model paths
    # bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv stablelm --mode calc_only
    bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv stablelm
done

for dataset in "${DATASETS[@]}"; do
    # Concatenate all model paths
    model_paths_arg=""
    for model_path in "${MODEL_PATHS[@]}"; do
        model_paths_arg+=" $model_path"
    done

    # Run the eval script with concatenated model paths
    bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv stablelm --mode calc_only
    # bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv stablelm
done