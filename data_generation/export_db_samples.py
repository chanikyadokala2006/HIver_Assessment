import pandas as pd
from sqlalchemy import create_engine
import os

def main():
    print("Connecting to PostgreSQL...")
    db_uri = 'postgresql://postgres:6264@localhost:5432/customer_support'
    engine = create_engine(db_uri)

    # We want 250 random inbound tweets for AppleSupport
    # We will export the User_Tweet and Company_Reply, and add empty columns for the user to hand-label
    query = """
    SELECT "User_Tweet", "Company_Reply"
    FROM conversations
    WHERE "Company" = 'AppleSupport'
    ORDER BY RANDOM()
    LIMIT 250;
    """
    
    print("Fetching 250 random rows for AppleSupport...")
    df = pd.read_sql(query, engine)
    
    # Add empty columns for hand-labeling
    df['Expected_Intent'] = ""
    df['Expected_Action (auto_handle/escalate)'] = ""
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'golden_dataset_unlabelled.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully exported {len(df)} rows to: {output_file}")
    print("You can now open this CSV in Excel or Google Sheets and manually fill in the empty columns to create your hand-labelled dataset!")

if __name__ == "__main__":
    main()
