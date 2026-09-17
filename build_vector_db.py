import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# LangChain and AI imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Build Chroma DB vector store for historical customer support replies.")
    parser.add_argument("--brand", type=str, default="AppleSupport", help="Brand name (default: AppleSupport)")
    args = parser.parse_args()
    
    BRAND_NAME = args.brand
    PERSIST_DIR = f"./chroma_db_{BRAND_NAME.lower()}"
    
    print(f"Building Persistent Vector DB for {BRAND_NAME}...")
    
    # 1. Load the top most frequently asked questions
    print(f"Fetching {BRAND_NAME} questions from local CSV...")
    
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        print("dataset.csv not found! Please ensure it is in the repository.")
        return
        
    # We group by Cleaned_User_Tweet to find the most repeated/common issues
    # Filter for the specific brand
    df = df[df['Company'] == BRAND_NAME]
    
    if df.empty:
        print(f"No data found for brand '{BRAND_NAME}' in dataset.csv!")
        return
        
    # Emulate the SQL grouping logic
    df = df.groupby('Cleaned_User_Tweet').agg(
        User_Tweet=('User_Tweet', 'first'),
        Company_Reply=('Company_Reply', 'first'),
        Cleaned_Company_Reply=('Cleaned_Company_Reply', 'first'),
        freq=('Cleaned_User_Tweet', 'count')
    ).reset_index().sort_values(by='freq', ascending=False)
        
    print(f"Loaded {len(df)} distinct conversations.")

    # 2. Setup Vector Store with local HuggingFace embeddings
    print("Initializing HuggingFace local embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        collection_name=f"{BRAND_NAME}_support".lower(),
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    # 3. Embed and Persist all at once! (Zero rate limits locally)
    print(f"Embedding {len(df)} items locally into Vector DB...")
    
    docs = df["Cleaned_User_Tweet"].tolist()
    metadatas = [
        {
            "cleaned_reply": row["Cleaned_Company_Reply"], 
            "raw_reply": row["Company_Reply"],
            "frequency": row["freq"]
        } 
        for _, row in df.iterrows()
    ]
    
    # Add to Chroma in chunks to avoid SQLite max batch size limits (max is 5461)
    chunk_size = 5000
    for i in range(0, len(docs), chunk_size):
        print(f"Adding chunk {i // chunk_size + 1} / {(len(docs) + chunk_size - 1) // chunk_size}...")
        chunk_docs = docs[i:i + chunk_size]
        chunk_metadatas = metadatas[i:i + chunk_size]
        vectorstore.add_texts(texts=chunk_docs, metadatas=chunk_metadatas)
        
    print(f"Successfully built and persisted Vector DB at '{PERSIST_DIR}'!")

if __name__ == "__main__":
    main()
