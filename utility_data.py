import json
import openai
from typing import List, Dict
import os
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

def filter_utility_rfps(data: List[Dict], api_key: str) -> None:
    """
    Filter and save utility industry RFPs to JSONL file
    """
    client = openai.Client(api_key=api_key)
    
    with open('datasets/utility_rfps.jsonl', 'w', encoding='utf-8') as f:
        for record in data:
            if len(record.get("description", "")) > 0:
                if is_utility_industry(record["description"], client):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

# Usage
load_dotenv() 

# Use os.environ.get function to extract credentials from .env file
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
# Use create_client function to connect to Supbase using credentials and then create a 'supbase' object
supabase: Client = create_client(url, key)
# Store all the columns of 'data' table in Supabase into a 'response' object
response = supabase.table("data").select("*").execute()
# Only extract the list of data where all the rows stored into each dictionary and then store them into a 'data' list
data = response.data
api_key = os.environ.get("OPENAI_API_KEY")
filter_utility_rfps(data, api_key)