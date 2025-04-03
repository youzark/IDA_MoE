MODEL_PATHS=(
    "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2/llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2/llavaqwen2-1.5b-finetune-moe-gmm-1.0-expert-16-center-16-dim-32"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2/llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-32-Bunny"
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2/llavaqwen2-1.5b-finetune-moe-gmm-1.0-center-16-dim-64"
)
    # "/apdcephfs_nj7/share_1273717/yannhua/Home/moe/exps/qwen2/"

# List of datasets
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa" "mmbench" "mmvet" "vizwiz" "vqav2")
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa" )
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa" )
# DATASETS=("mmbench")
# DATASETS=("mmvet")
# DATASETS=("mmbench-cn")
# DATASETS=("llavabench")
DATASETS=("mme")
# DATASETS=("mme" "pope" "sqa" "gqa" "textvqa")
# DATASETS=("pope")
# DATASETS=("vizwiz")
# DATASETS=("vqav2")

cd "/apdcephfs_nj7/share_1273717/yannhua/Home/moe"
# Loop through each dataset

for dataset in "${DATASETS[@]}"; do
    # Concatenate all model paths
    model_paths_arg=""
    for model_path in "${MODEL_PATHS[@]}"; do
        model_paths_arg+=" $model_path"
    done

    # Run the eval script with concatenated model paths
    bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv qwen
done

for dataset in "${DATASETS[@]}"; do
    # Concatenate all model paths
    model_paths_arg=""
    for model_path in "${MODEL_PATHS[@]}"; do
        model_paths_arg+=" $model_path"
    done

    # Run the eval script with concatenated model paths
    bash ./scripts/compare_llava_moe/eval/eval.sh --moe --model_paths $model_paths_arg --datasets "$dataset" --conv qwen --mode calc_only
done