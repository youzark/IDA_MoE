#!/bin/bash

function generate_eval_string() {
    local base_dirs=($1)    # Space-separated list of base directories
    local model_names=($2)  # Space-separated list of model names
    local datasets=($3)     # Space-separated list of dataset names
    local config_string=""
    
    # Validate input arrays have same length
    if [ ${#base_dirs[@]} -ne ${#model_names[@]} ]; then
        echo "Error: Number of base directories (${#base_dirs[@]}) must match number of model names (${#model_names[@]})" >&2
        return 1
    fi
    
    # Generate config string for each combination
    for i in "${!base_dirs[@]}"; do
        local base_dir="${base_dirs[$i]}"
        local model_name="${model_names[$i]}"
        
        for dataset in "${datasets[@]}"; do
            if [ -n "$config_string" ]; then
                config_string+="\n"
            fi
            config_string+="${base_dir}|${model_name}|${dataset}"
        done
    done
    
    echo "$config_string"
}

function usage() {
    echo "Usage: $0 --base_dir <dir> --model_name <name> --datasets <dataset1> [dataset2 ...]" >&2
    echo "Options:" >&2
    echo "  --base_dir   : Base directory path" >&2
    echo "  --model_name : Model name" >&2
    echo "  --datasets   : One or more dataset names" >&2
    exit 1
}

# Parse command line arguments
base_dir=""
model_name=""
datasets=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir)
            base_dir="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$base_dir" ]] || [[ -z "$model_name" ]] || [[ ${#datasets[@]} -eq 0 ]]; then
    echo "Error: Missing required arguments" >&2
    usage
fi

# Generate and output eval string
eval_string=$(generate_eval_string "$base_dir" "$model_name" "${datasets[*]}")
echo "$eval_string"
