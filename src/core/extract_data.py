import os
import json
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', 
                       type=str, 
                       default='../../datasets/data.jsonl',
                       help='Path to save the output JSONL file')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    # Initialize Supabase client
    supabase: Client = create_client(url, key)
    
    # Fetch data from Supabase
    response = supabase.table("data").select("*").execute()
    data = response.data
    
    # Save data to JSONL file
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for record in data:
            if len(record["description"]) > 0:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
