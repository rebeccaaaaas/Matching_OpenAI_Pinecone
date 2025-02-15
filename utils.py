import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client, Client
import tiktoken


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def get_embeddings_batch(texts, model="text-embedding-3-large"):
    encoding = tiktoken.get_encoding("cl100k_base")
    texts = [''.join(encoding.decode(encoding.encode(text)[:8192])) for text in texts]
    response = client.embeddings.create(input=texts, model=model).data
    embeddings = [res.embedding for res in response]
    return embeddings

def format_to_vector_dict(vector_id: str, values: list, metadata: dict):
    return {
        "id": vector_id,
        "values": values,
        "metadata": metadata,
    }

def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=[text], model=model).data
    response = response[0].embedding
    return response

def retrieve_top_k_similar_docs(question, index, namespace, k=3):
    encoding = tiktoken.get_encoding("cl100k_base")
    question_chunked = ''.join(encoding.decode(encoding.encode(question)[:8192]))
    question_embedding = get_embedding(question_chunked)
    related_data = index.query(vector=question_embedding, namespace=namespace, include_metadata=True, top_k=k)
    
    return related_data["matches"]