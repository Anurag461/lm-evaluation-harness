from lm_eval.evaluator import simple_evaluate
import json
import argparse  

def test_escitask(model):  
    print(f"Testing {model} with evaluation framework...")  
    
    tasks = ["esci"] 
    
    results_exact = simple_evaluate(
            model=model, 
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            device="cpu",
            verbosity="INFO")
            
    print("Evaluation Results:")
    
    results = {
        model: results_exact["results"]["esci"]["exact_match,none"]
    }
    print(results)
    
    output_file = f"evaluation_results_{model}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Evaluate a model on the ESCI task.")
    parser.add_argument("model", type=str, help="The model to evaluate.")
    args = parser.parse_args()
    test_escitask(args.model)
