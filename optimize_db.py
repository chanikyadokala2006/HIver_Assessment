import pandas as pd
import spacy
import re
from sqlalchemy import create_engine
import time

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_and_extract(text):
    if not isinstance(text, str) or not text:
        return ""
        
    # Replace URLs with [URL]
    text = re.sub(r'http\S+', '[URL]', text)
    
    # Process text
    doc = nlp(text)
    
    words = []
    for token in doc:
        # Keep [URL] placeholder
        if token.text == '[URL]':
            words.append('[URL]')
            continue
            
        # Ignore mentions
        if token.text.startswith('@'):
            continue
            
        # Keep content-heavy words (nouns, verbs, proper nouns, adjectives) and drop stop words
        if token.pos_ in ["NOUN", "VERB", "PROPN", "ADJ"] and not token.is_stop:
            words.append(token.lemma_.lower())
            
    return " ".join(words)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Optimize database conversations for a specific brand.")
    parser.add_argument("--brand", type=str, default="AppleSupport", help="The brand name to optimize conversations for.")
    args = parser.parse_args()
    
    BRAND_NAME = args.brand
    DB_URI = 'postgresql://postgres:6264@localhost:5432/customer_support'
    
    print("Connecting to PostgreSQL...")
    engine = create_engine(DB_URI)
    
    print(f"Loading conversations for {BRAND_NAME}...")
    query = f"""
        SELECT "Company", "User_ID", "User_Tweet", "Company_Reply" 
        FROM conversations 
        WHERE "Company" = '{BRAND_NAME}'
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No data found!")
        return
        
    print(f"Found {len(df)} conversations. Processing with NLP pipeline (this will take a few minutes)...")
    
    start_time = time.time()
    
    # Apply NLP extraction
    df['Cleaned_User_Tweet'] = df['User_Tweet'].apply(clean_and_extract)
    df['Cleaned_Company_Reply'] = df['Company_Reply'].apply(clean_and_extract)
    
    elapsed = time.time() - start_time
    print(f"NLP processing complete in {elapsed:.2f} seconds.")
    
    # Optional: Filter out empty cleaned strings (cases where it was just an @mention)
    df = df[df['Cleaned_User_Tweet'].str.len() > 0]
    
    print("Saving optimized data to table 'conversations_optimized'...")
    df.to_sql('conversations_optimized', engine, if_exists='replace', index=False, chunksize=10000)
    
    print("Optimization complete!")

if __name__ == '__main__':
    main()
