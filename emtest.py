from lm_eval.evaluator import simple_evaluate
import json

def test_em():
    """Test the ExactModel using the evaluation framework"""
    print("Testing ExactModel with evaluation framework...")
    
    tasks = ["esci"] 
    
    results_exact = simple_evaluate(
            model="exact",  # Use our registered exact model
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO")

    results_substitute = simple_evaluate(
            model="substitute",  # Use our registered exact model
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO")

    results_complement = simple_evaluate(
            model="complement",  # Use our registered exact model
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO")

    results_irrelevant = simple_evaluate(
            model="irrelevant",  # Use our registered exact model
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO")
            
    print("Evaluation Results:")
    
    results = {
        "exact": results_exact["results"]["exact_match,none"],
        "substitute": results_substitute["results"]["exact_match,none"],
        "complement": results_complement["results"]["exact_match,none"],
        "irrelevant": results_irrelevant["results"]["exact_match,none"]
    }
    print(results)
    # Write results to JSON file
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")
    
    print("Evaluation test completed.\n")

print("hello world")
test_em()
