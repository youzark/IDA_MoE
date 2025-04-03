import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import os

def parse_sample_id(sample_id):
    """Parse dataset name and data id from sample_id string."""
    try:
        dataset_name, data_id = sample_id.split('--')
        return dataset_name, data_id
    except ValueError:
        raise ValueError(f"Invalid sample_id format: {sample_id}")

def count_lines(file_path):
    """Count lines in a file efficiently."""
    def _raw_gen():
        with open(file_path, 'rb') as f:
            while chunk := f.read(1024*1024):
                yield chunk.count(b'\n')
    return sum(_raw_gen())

def load_routing_decisions(file_path):
    """Load routing decisions from jsonl file into a dictionary with progress bar."""
    decisions_dict = {}
    datasets = set()
    
    # Get total lines for progress bar
    print(f"Counting lines in {os.path.basename(file_path)}...")
    total_lines = count_lines(file_path)
    
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Loading decisions"):
            data = json.loads(line.strip())
            sample_id = data['sample_ids']
            dataset_name, data_id = parse_sample_id(sample_id)
            datasets.add(dataset_name)
            decisions_dict[sample_id] = data['routing_decision']
    
    return decisions_dict, datasets

def compare_decisions(decisions1, decisions2):
    """Compare two lists of routing decisions and return match percentage."""
    if len(decisions1) != len(decisions2):
        raise ValueError("Routing decisions have different lengths")
    
    total_matches = 0
    total_items = 0
    
    for expert_decisions1, expert_decisions2 in zip(decisions1, decisions2):
        if len(expert_decisions1) != len(expert_decisions2):
            raise ValueError("Expert decisions have different lengths")
        
        matches = sum(1 for x, y in zip(expert_decisions1, expert_decisions2) if x == y)
        total_matches += matches
        total_items += len(expert_decisions1)
    
    return (total_matches / total_items) * 100 if total_items > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Compare routing decisions between two files')
    parser.add_argument('file1', help='Path to first routing decision file')
    parser.add_argument('file2', help='Path to second routing decision file')
    args = parser.parse_args()

    # Load decisions from both files
    decisions1, datasets1 = load_routing_decisions(args.file1)
    decisions2, datasets2 = load_routing_decisions(args.file2)

    # Check if all samples exist in both files
    samples1 = set(decisions1.keys())
    samples2 = set(decisions2.keys())
    
    missing_in_2 = samples1 - samples2
    missing_in_1 = samples2 - samples1
    
    if missing_in_1 or missing_in_2:
        raise Exception(
            f"Samples missing in file2: {missing_in_2}\n"
            f"Samples missing in file1: {missing_in_1}"
        )

    # Calculate match percentage per dataset
    dataset_matches = defaultdict(list)
    
    print("Comparing decisions...")
    for sample_id in tqdm(samples1, desc="Comparing samples"):
        dataset_name, _ = parse_sample_id(sample_id)
        match_percentage = compare_decisions(
            decisions1[sample_id],
            decisions2[sample_id]
        )
        dataset_matches[dataset_name].append(match_percentage)

    # Calculate and print average match percentage per dataset
    print("\nResults per dataset:")
    print("-" * 50)
    for dataset_name, percentages in dataset_matches.items():
        avg_percentage = sum(percentages) / len(percentages)
        print(f"{dataset_name}: {avg_percentage:.2f}% match")

if __name__ == "__main__":
    main()