import os
import argparse
import json
import re
from datetime import datetime
from moellava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--output-dir', type=str, default='./results')
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(result_file))[0]
    print(model_name)
    
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    detailed_results = []
    
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred = {
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        }
        pred_list.append(pred)
        
        # Store detailed result for each question
        detailed_results.append({
            "question_id": result['question_id'],
            "question": annotation['question'],
            "prediction": result['text'],
            "ground_truth": annotation['answers']
        })

    evaluator = TextVQAAccuracyEvaluator()
    accuracy = evaluator.eval_pred_list(pred_list)
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * accuracy))

    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare summary results
    summary_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "dataset": "textvqa",
        "scores": {
            "accuracy": float(accuracy * 100),
            "num_samples": len(pred_list)
        }
    }

    # Prepare detailed results
    detailed_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "overall_accuracy": float(accuracy * 100),
        "num_samples": len(pred_list),
        "questions": detailed_results
    }

    # Save summary results
    summary_output = os.path.join(output_dir, f"{model_name}_textvqa_summary_{timestamp}.json")
    with open(summary_output, 'w') as f:
        json.dump(summary_dict, f, indent=2)

    # Save detailed results
    detailed_output = os.path.join(output_dir, f"{model_name}_textvqa_detailed_{timestamp}.json")
    with open(detailed_output, 'w') as f:
        json.dump(detailed_dict, f, indent=2)

    print(f"Results saved to: {output_dir}/")
    return summary_dict, detailed_dict


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args.output_dir)

    if args.result_dir is not None:
        all_results = []
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            summary, details = eval_single(
                args.annotation_file, 
                os.path.join(args.result_dir, result_file),
                args.output_dir
            )
            all_results.append(summary)
        
        # Save combined results if processing multiple files
        if all_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_output = os.path.join(args.output_dir, f"textvqa_combined_summary_{timestamp}.json")
            with open(combined_output, 'w') as f:
                json.dump(all_results, f, indent=2)
