import time
import argparse
import os
import sys

from data_loader import load_and_transform_dataset
from apriori import ProbabilisticApriori

def main():
    parser = argparse.ArgumentParser(description="Probabilistic Apriori Mining Algorithm")
    parser.add_argument("--csv", type=str, default="Groceries_dataset.csv", 
                        help="Path to the standard Groceries dataset CSV")
    parser.add_argument("--min_sup_ratio", type=float, default=0.001, 
                        help="Minimum support threshold ratio (e.g. 0.001 for 0.1%)")
    parser.add_argument("--min_conf", type=float, default=0.10, 
                        help="Minimum confidence threshold (e.g. 0.10 for 10%)")
    parser.add_argument("--deterministic", action="store_true", 
                        help="Disable uncertainty simulation and use deterministic values")
    
    args = parser.parse_args()
    
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: Dataset not found at '{csv_path}'.")
        print("Please check the path or download from: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset")
        sys.exit(1)
        
    try:
        uncertain_transactions = load_and_transform_dataset(
            csv_path, simulate_uncertainty=not args.deterministic
        )
        
        total_transactions = len(uncertain_transactions)
        if total_transactions == 0:
            print("No transactions found. Exiting.")
            sys.exit(0)
            
        min_support_threshold = total_transactions * args.min_sup_ratio
        
        print(f"\nProcessing {total_transactions} transactions...")
        print(f"Absolute Minimum Support count required: {min_support_threshold:.2f}\n")
        
        start_time = time.time()
        
        p_apriori = ProbabilisticApriori(
            min_support=min_support_threshold, 
            min_confidence=args.min_conf
        )
        p_apriori.fit(uncertain_transactions)
        
        probabilistic_rules = p_apriori.generate_rules()
        execution_time = time.time() - start_time
        
        print(f"\n========================================================")
        print(f"Mining Operation Concluded.")
        print(f"Total Execution Time: {execution_time:.3f} seconds")
        print(f"Total Probabilistic Rules Generated: {len(probabilistic_rules)}")
        print(f"========================================================")
        
        if probabilistic_rules:
            probabilistic_rules.sort(key=lambda x: (x['expected_confidence'], x['expected_support']), reverse=True)
            
            print(f"\nTop 10 Probabilistic Association Rules (Filtered by Expected Confidence):")
            for idx, rule in enumerate(probabilistic_rules[:10]):
                ant = ", ".join(rule['antecedent'])
                con = ", ".join(rule['consequent'])
                print(f"Rule {idx+1}: {{{ant}}} => {{{con}}}")
                print(f"    -> Expected Support:    {rule['expected_support']:.2f}")
                print(f"    -> Expected Confidence: {rule['expected_confidence']:.4f}")
                print(f"    -> Probabilistic Lift:  {rule['lift']:.4f}\n")
        else:
            print("\nNo rules generated. Try lowering the minimum support or confidence thresholds.")
            
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()