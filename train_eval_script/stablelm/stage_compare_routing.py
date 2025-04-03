import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def parse_sample_id(sample_id):
    """Parse dataset name and data id from sample_id string."""
    try:
        dataset_name, data_id = sample_id.split('--')
        return dataset_name, data_id
    except ValueError:
        raise ValueError(f"Invalid sample_id format: {sample_id}")

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

def process_files_in_steps(file1_path, file2_path, step_size):
    """Process files in steps and calculate match percentages."""
    step_matches = defaultdict(list)  # {dataset_name: [(step_num, match_percentage), ...]}
    current_step = 0
    step_data = defaultdict(list)  # Temporary storage for current step's data
    
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        pbar = tqdm(desc="Processing lines")
        while True:
            # Read lines from both files
            line1 = f1.readline()
            line2 = f2.readline()
            
            # Check if we've reached the end of either file
            if not line1 or not line2:
                if line1 != line2:  # One file is longer than the other
                    raise Exception("Files have different number of lines")
                break
            
            # Parse the lines
            data1 = json.loads(line1.strip())
            data2 = json.loads(line2.strip())
            
            # Verify sample IDs match
            if data1['sample_ids'] != data2['sample_ids']:
                raise Exception(f"Mismatched sample IDs at line {pbar.n + 1}")
            
            dataset_name, _ = parse_sample_id(data1['sample_ids'])
            match_percentage = compare_decisions(
                data1['routing_decision'],
                data2['routing_decision']
            )
            
            # Add to current step's data
            step_data[dataset_name].append(match_percentage)
            
            # If we've reached the step size, calculate averages and reset
            if (pbar.n + 1) % step_size == 0:
                step_num = current_step
                # Calculate averages for each dataset in this step
                for dataset, percentages in step_data.items():
                    avg_percentage = sum(percentages) / len(percentages)
                    step_matches[dataset].append((step_num, avg_percentage))
                
                # Reset step data
                step_data.clear()
                current_step += 1
            
            pbar.update(1)
        
        # Handle any remaining data in the last step
        if step_data:
            step_num = current_step
            for dataset, percentages in step_data.items():
                avg_percentage = sum(percentages) / len(percentages)
                step_matches[dataset].append((step_num, avg_percentage))

    return step_matches

def main():
    parser = argparse.ArgumentParser(description='Compare routing decisions between two files in steps')
    parser.add_argument('file1', help='Path to first routing decision file')
    parser.add_argument('file2', help='Path to second routing decision file')
    parser.add_argument('--step-size', type=int, default=10000, help='Number of lines per step')
    args = parser.parse_args()

    # Process files and get results
    step_matches = process_files_in_steps(args.file1, args.file2, args.step_size)

    # Print results
    print("\nResults per dataset and step:")
    print("-" * 50)
    for dataset_name, step_results in step_matches.items():
        print(f"\nDataset: {dataset_name}")
        print("Step\tMatch Percentage")
        print("-" * 25)
        for step_num, percentage in step_results:
            start_line = step_num * args.step_size
            end_line = (step_num + 1) * args.step_size - 1
            print(f"{start_line}-{end_line}:\t{percentage:.2f}%")
        
        # Calculate overall average for this dataset
        overall_avg = sum(pct for _, pct in step_results) / len(step_results)
        print(f"Overall average: {overall_avg:.2f}%")

if __name__ == "__main__":
    main()