import json
import openai
from typing import List, Dict
import os
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

def is_utility_industry(description: str, client: openai.Client) -> bool:
    """
    Check if the description is related to utility industry using OpenAI API
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier that determines if a RFP description is related to the utility industry (electricity, water, gas, etc.). Reply with only 'yes' or 'no'."
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content.lower().strip() == 'yes'
    except Exception as e:
        print(f"Error classifying description: {e}")
        return False

def filter_utility_rfps(data: List[Dict], api_key: str, output_path: str) -> None:
    """
    Filter and save utility industry RFPs to JSONL file
    """
    client = openai.Client(api_key=api_key)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            if len(record.get("description", "")) > 0:
                if is_utility_industry(record["description"], client):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', 
                       type=str, 
                       default='../../datasets/utility_rfps.jsonl',
                       help='Path to save the filtered utility RFPs')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize Supabase client
    supabase: Client = create_client(url, key)
    
    # Fetch data from Supabase
    response = supabase.table("data").select("*").execute()
    data = response.data
    
    # Filter and save utility RFPs
    filter_utility_rfps(data, api_key, args.output_path)

if __name__ == "__main__":
    main()