import os
import argparse
import pandas as pd
from agent import SupportAgent
from rouge_score import rouge_scorer
import time
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Golden Dataset Evaluation")
    parser.add_argument("--csv", type=str, default="data_generation/golden_dataset_labelled.csv.csv", help="Path to the hand-labelled CSV")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to evaluate")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        logging.error(f"Could not find golden dataset at {args.csv}")
        return

    print("="*80)
    print(f" GOLDEN DATASET EVALUATION HARNESS")
    print("="*80)
    
    # Load the CSV
    df = pd.read_csv(args.csv)
    
    # Limit to the requested number of examples (making sure we don't exceed the dataframe size)
    limit = min(args.limit, len(df))
    df = df.head(limit)
    
    print(f"Loaded {limit} labelled examples from {args.csv}\n")
    print("Initializing Support Agent...")
    agent = SupportAgent(brand_name="AppleSupport")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    correct_intent = 0
    correct_action = 0
    total_rouge_l = 0.0
    
    for idx, row in df.iterrows():
        user_tweet = row['User_Tweet']
        ground_truth_reply = row['Company_Reply']
        expected_intent = str(row['Expected_Intent']).strip().lower()
        expected_action = str(row['Expected_Action (auto_handle/escalate)']).strip().lower()
        
        print("-" * 80)
        print(f"Example {idx+1}/{limit}")
        safe_tweet = str(user_tweet).encode('ascii', 'replace').decode('ascii')
        safe_gt = str(ground_truth_reply).encode('ascii', 'replace').decode('ascii')
        print(f"User Tweet: {safe_tweet}")
        print(f"Ground Truth Reply: {safe_gt}")
        print(f"Expected Intent: {expected_intent}")
        print(f"Expected Action: {expected_action}")
        print("...\nThinking...")
        
        # Run Agent
        prediction, _ = agent.handle_message(user_tweet)
        
        # Calculate Metrics
        predicted_intent = prediction.intent.strip().lower()
        predicted_action = "auto_handle" if prediction.auto_handle else "escalate"
        rouge_scores = scorer.score(ground_truth_reply, prediction.draft_reply)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # Track accuracy
        if predicted_intent == expected_intent: correct_intent += 1
        if predicted_action == expected_action: correct_action += 1
        total_rouge_l += rouge_l
        
        print(f"\n[AI PREDICTION]")
        print(f"Intent: {prediction.intent} | Action: {predicted_action}")
        print(f"Draft Reply: {prediction.draft_reply}")
        print(f"ROUGE-L Score: {rouge_l:.4f}")
        
        print("\nWaiting 15s to respect API rate limits...")
        time.sleep(15) # Wait to avoid API limit (5 requests per minute)
        
    print("="*80)
    print(" GOLDEN DATASET METRICS SUMMARY")
    print("="*80)
    print(f"Total Evaluated: {limit}")
    print(f"Intent Accuracy: {(correct_intent/limit)*100:.1f}% ({correct_intent}/{limit})")
    print(f"Action (Routing) Accuracy: {(correct_action/limit)*100:.1f}% ({correct_action}/{limit})")
    print(f"Average ROUGE-L: {total_rouge_l/limit:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
