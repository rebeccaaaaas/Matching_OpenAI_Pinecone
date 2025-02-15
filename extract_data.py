import os
from supabase import create_client, Client
from dotenv import load_dotenv # Store Supabase URL and API Key in .env file and then load the file
import json # Store extracted Supabase data into JSON file

# Load .env file where Supbase URL and API key stored. 
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
# Open a new 'data.jsonl' file with write mode and only convert dictionaries with 'description' into a JSON string
with open('datasets/data.jsonl', 'w', encoding='utf-8') as f:
    for record in data:
        if len(record["description"]) > 0:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
