import pandas as pd
from sqlalchemy import create_engine
import os

def main():
    csv_path = os.path.join('archive', 'twcs', 'twcs.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find dataset at {csv_path}")
        return

    print("Loading data...")
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Original Data shape: {df.shape}")

    print("Separating user tweets and company replies...")
    # Separate the dataset into user tweets (inbound) and company replies (outbound)
    user_tweets = df[df['inbound'] == True]
    company_tweets = df[df['inbound'] == False]

    print("Merging tweets to link user's tweet with company's direct reply...")
    # Merge the tweets to link the user's tweet with the company's direct reply
    conversations = pd.merge(
        user_tweets, 
        company_tweets, 
        left_on='tweet_id', 
        right_on='in_response_to_tweet_id', 
        suffixes=('_user', '_company')
    )

    print("Filtering and renaming columns...")
    # Keep only the relevant columns for readability
    conversations = conversations[['author_id_company', 'author_id_user', 'text_user', 'text_company']]

    # Rename the columns so they are easier to understand
    conversations.columns = ['Company', 'User_ID', 'User_Tweet', 'Company_Reply']

    print("Sorting by Company...")
    # Sort the dataframe by the Company name to group them together
    conversations_grouped = conversations.sort_values(by='Company').reset_index(drop=True)

    print(f"Final shape of grouped data: {conversations_grouped.shape}")
    print(conversations_grouped.head())

    print("\nConnecting to PostgreSQL...")
    # Connection string for the docker-compose setup
    # postgresql://user:password@host:port/database_name
    db_uri = 'postgresql://postgres:6264@localhost:5432/customer_support'
    engine = create_engine(db_uri)

    print("Writing data to database table 'conversations' (this might take a while)...")
    # Write to SQL in chunks to manage memory
    conversations_grouped.to_sql('conversations', engine, if_exists='replace', index=False, chunksize=10000)

    print("Data successfully stored in PostgreSQL database!")

if __name__ == '__main__':
    main()
