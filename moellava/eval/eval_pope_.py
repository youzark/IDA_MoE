import os
import json
import argparse
from datetime import datetime

def compute_metrics(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
    
    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "yes_ratio": yes_ratio
    }

def process_answers(answers):
    processed_answers = []
    for answer in answers:
        text = answer['text']
        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            processed_answers.append(0)  # no
        else:
            processed_answers.append(1)  # yes
    return processed_answers

def eval_pope(answers, label_file):
    # More robust label loading
    label_list = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    label = json.loads(line)['label']
                    label_list.append(0 if label == 'no' else 1)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {label_file}: {line}")
                    raise e
    
    pred_list = process_answers(answers)
    return compute_metrics(pred_list, label_list)
# def eval_pope(answers, label_file):
#     label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
#     label_list = [0 if label == 'no' else 1 for label in label_list]
#     pred_list = process_answers(answers)
#     return compute_metrics(pred_list, label_list)

def evaluate_pope(args):
    # Extract model name from result file path
    model_name = os.path.basename(args.result_file).replace('.jsonl', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process questions and answers
    questions = {q['question_id']: q for q in [json.loads(line) for line in open(args.question_file)]}
    answers = [json.loads(q) for q in open(args.result_file)]
    
    result_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "categories": {}
    }
    
    total_f1 = 0
    num_categories = 0
    
    print("\n=========== POPE Categories ===========")
    for file in sorted(os.listdir(args.annotation_dir)):
        if not (file.startswith('coco_pope_') and file.endswith('.json')):
            continue
            
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        metrics = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        
        # Use F1 score * 100 as the category score
        category_f1 = metrics["f1"] * 100
        total_f1 += category_f1
        num_categories += 1
        
        result_dict["categories"][category] = {
            "f1_score": float(category_f1),
            "metrics": metrics,
            "num_samples": len(cur_answers)
        }
        
        print(f"\t {category}  score: {category_f1:.2f}")
        print(f"\t\tTP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
        print(f"\t\tPrecision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
    
    print(f"total score: {total_f1}")
    result_dict["overall_f1"] = total_f1
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_output = os.path.join(args.output_dir, f"{model_name}_pope_detailed_{timestamp}.json")
    with open(detailed_output, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Save summary results
    summary_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "dataset": "pope",
        "scores": {
            "overall_f1": float(total_f1),
            "average_f1": float(total_f1/num_categories)
        }
    }
    
    summary_output = os.path.join(args.output_dir, f"{model_name}_pope_summary_{timestamp}.json")
    with open(summary_output, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}/")
    
    return summary_dict, result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    
    summary, details = evaluate_pope(args)
