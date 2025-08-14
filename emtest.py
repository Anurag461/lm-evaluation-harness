import json

def test_em():
    """Test the ExactModel using the evaluation framework"""
    print("Testing ExactModel with evaluation framework...")
    
    # TODO: Replace with actual toy dataset
    toy_tasks = ["esci"]  # Placeholder for toy dataset
    
    results = simple_evaluate(
            model="exact",  # Use our registered exact model
            tasks=toy_tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO",
            limit=100
        )
        
    print("Evaluation Results:")
    
    # Print out the metrics
    if 'results' in results:
        for task_name, task_results in results['results'].items():
            print(f"\nTask: {task_name}")
            assert task_results["exact_match,none"]=0.55
            for metric_name, metric_value in task_results.items():
                print(f"  {metric_name}: {metric_value}")
    
    # Write results to JSON file
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")
    
    print("Evaluation test completed.\n")

print("hello world")
test_em()
