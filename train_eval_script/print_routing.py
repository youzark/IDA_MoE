import json

def analyze_routing_decisions(file_path):
    results = {}

    # Read all IDs
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            routing_decision = data['routing_decision']
            sample_id = int(data["sample_ids"].split("--")[1])
            dataset = data["sample_ids"].split("--")[0]
            shape = len(routing_decision[0]) - 1
            if dataset not in results:
                results[dataset] = []
            results[dataset].append((sample_id, shape))

    for key in results:
        print(key)
    # Sort by sample_id
        sorted_results = sorted(results[key], key=lambda x: x[0])

        # Get all sample IDs
        sample_ids = [x[0] for x in sorted_results]

     # Check range completeness
        expected_ids = set(range(10000))  # 0 to 999
        actual_ids = set(sample_ids)

    # Print sorted results
        for sample_id, shape in sorted_results:
            print(f"Sample ID: {sample_id}, Shape: {shape}")


# Example usage
analyze_routing_decisions('routing_decision.jsonl')
