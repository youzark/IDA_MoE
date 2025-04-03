import json

def analyze_routing_decisions(file_path):
    results = []

    # Read all IDs
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            routing_decision = data['routing_decision']
            sample_id = data["sample_ids"]
            shape = len(routing_decision[0]) - 1
            results.append((sample_id, shape))

    # Sort by sample_id
    sorted_results = sorted(results, key=lambda x: x[0])

    # Get all sample IDs
    sample_ids = [x[0] for x in sorted_results]

    # Check range completeness
    expected_ids = set(range(10000))  # 0 to 999
    actual_ids = set(sample_ids)

    # Print sorted results
    for sample_id, shape in sorted_results:
        print(f"Sample ID: {sample_id}, Shape: {shape}")

    # Find missing IDs
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids

    print(f"Total samples found: {len(sample_ids)}")
    if len(missing_ids) == 0:
        print("Range is complete! All IDs from 0 to 999 are present.")
    else:
        print(f"Missing {len(missing_ids)} IDs:")
        print(f"Missing IDs: {sorted(list(missing_ids))}")

    if len(extra_ids) > 0:
        print(f"\nFound {len(extra_ids)} unexpected IDs outside range 0-999:")
        print(f"Extra IDs: {sorted(list(extra_ids))}")

# Example usage
analyze_routing_decisions('routing_decision.jsonl')
