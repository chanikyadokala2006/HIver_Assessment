import os
import argparse
import pandas as pd
from agent import SupportAgent
from rouge_score import rouge_scorer
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import time
import logging
import warnings
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

# Suppress the automatic function calling warning from LangChain
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

load_dotenv()

class LLMJudgeResult(BaseModel):
    score: int = Field(description="Score from 1 to 5 rating the overall performance of the AI Support Agent (Intent Classification, Escalation Decision, and Draft Reply).")
    reasoning: str = Field(description="Detailed reasoning for the score covering intent classification, escalation decision, and draft reply quality.")

def create_judge():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    ).with_structured_output(LLMJudgeResult)
    return llm

def main():
    parser = argparse.ArgumentParser(description="AI Support Agent Evaluation Harness")
    parser.add_argument("--brand", type=str, default="AppleSupport", help="Brand name to evaluate (default: AppleSupport)")
    parser.add_argument("--limit", type=int, default=5, help="Number of test conversations to evaluate (default: 5)")
    args = parser.parse_args()
    
    BRAND_NAME = args.brand
    
    print("="*80)
    print(f" AI SUPPORT AGENT EVALUATION HARNESS - Brand: {BRAND_NAME}")
    print("="*80)
    print("Setting up Agent and Judge...")
    agent = SupportAgent(brand_name=BRAND_NAME)
    judge_llm = create_judge()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print("Loading test data (Hold-out set) from dataset.csv...")
    try:
        df = pd.read_csv('dataset.csv')
        df = df[df['Company'] == BRAND_NAME]
        test_df = df[['User_Tweet', 'Company_Reply']].sample(n=min(args.limit, len(df)))
    except FileNotFoundError:
        print("dataset.csv not found! Please ensure it is in the repository.")
        return
    
    results = []
    
    print(f"\nEvaluating {len(test_df)} conversations for {BRAND_NAME}...\n")
    
    for idx, row in test_df.iterrows():
        user_msg = row['User_Tweet']
        ground_truth = row['Company_Reply']
        
        print("-" * 80)
        print(f"--- Conversation {idx+1}/{len(test_df)} ---")
        print(f"User Issue: {user_msg}\n")
        print(f"Ground Truth (Company): {ground_truth}\n")
        
        # 1. Generate Prediction
        try:
            agent_res, retrieved_context = agent.handle_message(user_msg)
            ai_reply = agent_res.draft_reply
            intent = agent_res.intent
            auto_handle = agent_res.auto_handle
            escalation_reason = agent_res.escalation_reason
        except Exception as e:
            print(f"Error calling agent: {e}")
            ai_reply = "ERROR"
            intent = "ERROR"
            auto_handle = False
            escalation_reason = "Error"
            retrieved_context = "ERROR"
            
        print(f"Agent Intent: {intent}")
        decision_str = "Auto-handle" if auto_handle else f"Escalate ({escalation_reason})"
        print(f"Agent Escalation Decision: {decision_str}")
        print(f"Agent Draft Reply: {ai_reply}\n")
            
        # 2. Compute ROUGE
        rouge_score = scorer.score(ground_truth, ai_reply)['rougeL'].fmeasure
        
        # 3. Call LLM Judge
        judge_prompt = f"""
        You are an incredibly strict, brutal, and unforgiving QA evaluator for customer support at {BRAND_NAME}.
        Your job is to scrutinize the AI Agent's ENTIRE performance (Intent Classification, Escalation Decision, and Draft Reply) against the User Issue and Ground Truth (historical company reply).
        
        INPUT DATA:
        - Brand: {BRAND_NAME}
        - User Issue: {user_msg}
        - Ground Truth (Historical Company Reply): {ground_truth}
        
        AGENT PERFORMANCE TO EVALUATE:
        - Classified Intent: {intent}
        - Escalation Decision: {decision_str}
        - AI Draft Reply: {ai_reply}
        
        EVALUATION CRITERIA:
        1. INTENT CLASSIFICATION: Is the intent accurate for the user's message?
        2. ESCALATION DECISION (MOST IMPORTANT): Did the AI make the right routing call? 
           - Auto-handling severe frustration/sensitive account issues is a major failure.
           - Unnecessarily escalating standard queries that have clear templates in the Ground Truth is also penalized.
        3. DRAFT REPLY: 
           - The draft reply exists primarily to show the agent's thought process.
           - DO NOT give a score of 1 just because the AI included a generic or placeholder URL (e.g. apple.com). It is NOT trained as a full customer service bot yet, just a router! 
           - The reply should just reflect a reasonable approximation of how a human might respond.
        
        Score from 1 to 5, where:
        1 = Severe failure: Wrong intent AND terrible escalation choice.
        2 = Poor performance: Bad escalation choice, though intent might be okay.
        3 = Acceptable: Gets the escalation and intent mostly right, but draft reply reasoning is off.
        4 = Good: Accurate intent and correct escalation decision.
        5 = Perfect: Spot-on intent, perfect escalation call, and the draft reply clearly aligns with the decision.
        """
        
        @retry(wait=wait_exponential(multiplier=1, min=10, max=60), stop=stop_after_attempt(10))
        def invoke_judge():
            return judge_llm.invoke(judge_prompt)
        
        try:
            judge_evaluation = invoke_judge()
            llm_score = judge_evaluation.score
            llm_reasoning = judge_evaluation.reasoning
        except Exception as e:
            print(f"Error calling judge: {e}")
            llm_score = 0
            llm_reasoning = "ERROR"
            
        print(f"LLM Judge Score: {llm_score} | ROUGE-L: {rouge_score:.4f}")
        print(f"LLM Reasoning: {llm_reasoning}\n")
        
        # 4. Human Evaluation
        human_score_input = input("Please rate the AI Draft Reply on a scale of 1-5 (or 'q' to quit): ")
        if human_score_input.lower() == 'q':
            print("Quitting early...")
            break
        
        try:
            human_score = int(human_score_input)
        except ValueError:
            human_score = 0
            
        results.append({
            "User_Tweet": user_msg,
            "Ground_Truth": ground_truth,
            "AI_Reply": ai_reply,
            "Intent": intent,
            "Auto_Handle": auto_handle,
            "Escalation_Reason": escalation_reason,
            "ROUGE_L": round(rouge_score, 4),
            "LLM_Score": llm_score,
            "LLM_Reasoning": llm_reasoning,
            "Human_Score": human_score
        })
        
        # Sleep slightly to avoid rate limit (Gemini Free Tier has 15 RPM limit)
        if idx < len(test_df) - 1:
            print("\nWaiting 10s to respect API rate limits...")
            time.sleep(10)
        
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv("evaluation_results.csv", index=False)
        print("\n" + "="*80)
        print(" EVALUATION COMPLETE")
        print("="*80)
        print("Saved to evaluation_results.csv")
        
        # Calculate means
        valid_scores = out_df[out_df["LLM_Score"] > 0]
        if not valid_scores.empty:
            print(f"Average ROUGE-L: {valid_scores['ROUGE_L'].mean():.4f}")
            print(f"Average LLM Score: {valid_scores['LLM_Score'].mean():.2f} / 5")
            
            # Calculate human agreement
            valid_human = out_df[out_df["Human_Score"] > 0]
            if not valid_human.empty:
                exact_match = (valid_human["LLM_Score"] == valid_human["Human_Score"]).mean()
                within_one = (abs(valid_human["LLM_Score"] - valid_human["Human_Score"]) <= 1).mean()
                print(f"Exact LLM-Human Agreement Rate: {exact_match*100:.1f}%")
                print(f"Agreement within 1 point: {within_one*100:.1f}%")

if __name__ == "__main__":
    main()
