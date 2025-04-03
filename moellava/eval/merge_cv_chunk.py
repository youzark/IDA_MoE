"""
Used to merge cv(measure the evenness of token allocation during inference) chunks
"""

import os
import argparse
import json

def merge_jsonl_files(input_folder, output_file):
    # with open(output_file, 'w') as outfile:
    cv_list = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.json') and 'chunk_cv' in filename:
            with open(os.path.join(input_folder, filename), 'r') as infile:
                cvs = json.load(infile)
                cv_list += cvs
    print("#"*100)
    print(f"{sum(cv_list)/len(cv_list)}")
    print("#"*100)
                
                # for line in infile:
                    # outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files into one.")
    parser.add_argument('--input-folder', type=str, required=True, help="Folder containing JSON files to merge.")
    parser.add_argument('--output-file', type=str, required=True, help="Output file for the merged JSON content.")
    
    args = parser.parse_args()
    
    merge_jsonl_files(args.input_folder, args.output_file)
