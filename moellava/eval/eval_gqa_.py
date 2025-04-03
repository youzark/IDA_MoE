import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

def compute_gqa_metrics(questions, predictions, scenes=None, choices=None, attentions=None, args=None):
    scores = {
        "accuracy": [],
        "binary": [],
        "open": [],
        "validity": [],
        "plausibility": [],
        "consistency": [],
        "accuracyPerStructuralType": defaultdict(list),
        "accuracyPerSemanticType": defaultdict(list),
        "accuracyPerLength": defaultdict(list),
        "accuracyPerSteps": defaultdict(list),
        "grounding": []
    }

    # Distribution tracking
    dist = {
        "gold": defaultdict(lambda: defaultdict(int)),
        "predicted": defaultdict(lambda: defaultdict(int))
    }

    def get_words_num(question):
        return len(question["question"].split())

    def get_steps_num(question):
        return len([c for c in question["semantic"] if not (any([o in "{}: {}".format(c["operation"], c["argument"])
                                                             for o in ["exist", "query: name", "choose name"]]))])

    # Process each question
    for qid, question in tqdm(questions.items()):
        if question["isBalanced"]:
            gold = question["answer"]
            predicted = predictions[qid]
            correct = (predicted == gold)
            score = float(correct)

            words_num = get_words_num(question)
            steps_num = get_steps_num(question)

            # Update accuracy metrics
            scores["accuracy"].append(score)
            scores["accuracyPerLength"][words_num].append(score)
            scores["accuracyPerSteps"][steps_num].append(score)
            scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
            scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
            
            answer_type = "open" if question["types"]["structural"] == "query" else "binary"
            scores[answer_type].append(score)

            # Update distribution tracking
            global_group = question["groups"]["global"]
            if global_group is not None:
                dist["gold"][global_group][gold] += 1
                dist["predicted"][global_group][predicted] += 1

    # Average scores
    results_dict = {
        "accuracy": sum(scores["accuracy"]) * 100.0 / len(scores["accuracy"]),
        "binary": sum(scores["binary"]) * 100.0 / len(scores["binary"]) if scores["binary"] else 0,
        "open": sum(scores["open"]) * 100.0 / len(scores["open"]) if scores["open"] else 0,
    }

    # Detailed scores per type
    detailed_scores = {}
    for score_type in ["accuracyPerStructuralType", "accuracyPerSemanticType", "accuracyPerLength", "accuracyPerSteps"]:
        detailed_scores[score_type] = {}
        for key, values in scores[score_type].items():
            if values:
                detailed_scores[score_type][key] = {
                    "score": sum(values) * 100.0 / len(values),
                    "count": len(values)
                }

    return results_dict, detailed_scores

def evaluate_gqa(args):
    # Load questions
    print("Loading questions...")
    questions = json.load(open(args.questions))

    # Load predictions
    print("Loading predictions...")
    predictions = json.load(open(args.predictions))
    predictions = {p["questionId"]: p["prediction"] for p in predictions}

    # Extract model name from predictions file path
    model_name = os.path.basename(args.predictions).replace('.json', '').replace('_predictions', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute metrics
    results_dict, detailed_scores = compute_gqa_metrics(questions, predictions)

    # Prepare summary results
    summary_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "dataset": "gqa",
        "scores": {
            "overall": float(results_dict["accuracy"]),
            "binary": float(results_dict["binary"]),
            "open": float(results_dict["open"])
        }
    }

    # Prepare detailed results
    detailed_dict = {
        "model_name": model_name,
        "timestamp": timestamp,
        "overall_scores": results_dict,
        "detailed_scores": detailed_scores
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save summary
    summary_output = os.path.join(args.output_dir, f"{model_name}_gqa_summary_{timestamp}.json")
    with open(summary_output, 'w') as f:
        json.dump(summary_dict, f, indent=2)

    # Save detailed results
    detailed_output = os.path.join(args.output_dir, f"{model_name}_gqa_detailed_{timestamp}.json")
    with open(detailed_output, 'w') as f:
        json.dump(detailed_dict, f, indent=2)

    # Print results
    print("\n=========== GQA Results ===========")
    print(f"Overall Accuracy: {results_dict['accuracy']:.2f}%")
    print(f"Binary Questions: {results_dict['binary']:.2f}%")
    print(f"Open Questions: {results_dict['open']:.2f}%")

    print("\nDetailed Accuracy by Types:")
    for type_name, type_scores in detailed_scores.items():
        print(f"\n{type_name}:")
        for key, values in type_scores.items():
            print(f"\t{key}: {values['score']:.2f}% ({values['count']} questions)")

    print(f"\nResults saved to: {args.output_dir}/")

    return summary_dict, detailed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save result files")
    args = parser.parse_args()

    summary, details = evaluate_gqa(args)
