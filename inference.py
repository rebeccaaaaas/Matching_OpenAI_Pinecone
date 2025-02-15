from utils import load_jsonl, retrieve_top_k_similar_docs
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
import json
from tqdm import tqdm

load_dotenv() # .env 변수 로드

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Using OPENAI GPT embedding function to covert descriptions in RFPs into numerical vectors
# Access OPENAI API key by using OPENAI_API_KEY
# Define OPENAI client which interacts with OPENAI API
client = OpenAI(api_key=OPENAI_API_KEY)
# Access PINECONE API by using PINECONE_API_KEY
# Define PINECONE client which interacts with PINECONE API
pc = Pinecone(api_key=PINECONE_API_KEY)
# Define a main storge(Index) as "rfp"
index = pc.Index("rfp")
# Define a namespace within the index as "openai-no-chunk"
namespace = "openai-no-chunk"
# Define the path where data.jsonl file is stored and stored the contents in a data object
data = load_jsonl("datasets/utility_rfps.jsonl")
# Define the path where test_skill_sets is stored and stored the skill sets in a skill_sets object 
skill_sets = load_jsonl("datasets/test_skill_sets.jsonl")
# Match the skill sets with RFPs descriptions and store sample results in a 'test_matched_docs' JSONL file 
save_path = "datasets/utest_matched_docs.jsonl"
save_path2 = "datasets/utest_matchescores.jsonl"

for idx, skill_set in enumerate(tqdm(skill_sets), start=1):
    # Load skill sets 
    skills = skill_set['text']
    # Used 'retrieve_top_k_similar_docs' to embed skill sets and compare them with vectorized descriptions using consine similarity search function
    # provided by PINCONE and extracted top 3 most similar matches 
    # retrieved_docs includes vector IDs and their consine similarity scores for the top 3 most similar matches 
    retrieved_docs = retrieve_top_k_similar_docs(skills, index, namespace, k=3)
    # 在处理文档之前添加
    # Create a 'test_matched_docs.jsonl' file and open the file 
    with open(save_path.replace('.jsonl', '.txt'), 'a') as f:
        f.write(f"{idx}. skill_set: {skills} 에 대한 top3 matched docs:\n")
        # Retrieved top 3 similarity results using vector id to look up the matched 'postingId' and 'description' in original dataset 
        for i, retrieved_doc in enumerate(retrieved_docs, start=1):
            similarity_score = retrieved_doc['score']
            posting_id = data[int(retrieved_doc['id'])]['postingId']
            description = data[int(retrieved_doc['id'])]['description']
            f.write(f"{idx}-{i}. postingId: {posting_id} \n similarity: {similarity_score:.4f}\n description: {description}\n\n")
        f.write("------------------------------------------------------------------------------------------------------------------------\n\n")
    with open(save_path2.replace('.jsonl', '.txt'), 'a') as f:
        f.write(f"{idx}. skill_set: {skills} 에 대한 top3 matched docs:\n")
        # Retrieved top 3 similarity results using vector id to look up the matched 'postingId' and 'description' in original dataset 
        for i, retrieved_doc in enumerate(retrieved_docs, start=1):
            similarity_score = retrieved_doc['score']
            posting_id = data[int(retrieved_doc['id'])]['postingId']
            description = data[int(retrieved_doc['id'])]['description']
            f.write(f"{idx}-{i}. postingId: {posting_id} \n similarity: {similarity_score:.4f}\n \n")
        f.write("------------------------------------------------------------------------------------------------------------------------\n\n")
