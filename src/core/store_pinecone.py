import os
from utils.utils import load_json, load_jsonl, get_embeddings_batch, format_to_vector_dict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm
import argparse

# To enhance the security, use .env file to load the variables without hard coding 
load_dotenv() # .env 변수 로드

# Use the 'environ' function to extract the variables from the .env file 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Creates an instance of the OpenAI class, using the credentials for authentication
# Interact with OpenAI's API like generating embeddings or using GPT models
client = OpenAI(api_key=OPENAI_API_KEY)

# Create an instance of the Pinecone class, using the credentials for authentication
# Initialize a connection to Pinecone using to store, query, and manage vector data
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use 'parser' object to pass required arguments to the program for embedding 
# Define input descriptions that are needed to parse the jsonl file and store them into pinecone
parser = argparse.ArgumentParser(description="Store to pinecone")
# Parse the data.jsonl file to prepare it for storage
# Define the 'index' argument to specify the exact location in which the numerical vectors store in pinecone
parser.add_argument('--index', type=str, required=True, help='index')
# Define the 'namespace' argument to group data within the same index
# The outcomes could be separated into different groups within the same index
parser.add_argument('--namespace', type=str, required=True, help='Namespace')
# Define the 'jsonl_path' to specify the exact location of the file on your local computer
parser.add_argument('--jsonl_path', type=str, required=True, help='jsonl')
# Store command-line arguments defined in 'parser' into 'args'
args = parser.parse_args()

# Embedding process for the texts in RPFs and store the outcomes of numerical vectors into Pinecone (a vector database)
# index : the location where numerical vectors are stored 
# chunks: list -> text chunks prepared for embedding *Converting JSON file into list  
# metadata_list includes other information execpt for 'description', the main source for embedding, such as 'PostingId'
# batch_size -> Set a maximum of 100 text chunks to be embedded at once (Converting text chunks into numerical vectors)
def store_vector_to_pinecone(index, chunks: list, metadata_list: list, namespace, batch_size=100):
# tqdm shows the progress bar of embedding process
# text chunks are divided into batches of 100 and each bach is embedded separately (0~99, 100~199, 200~299...)
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Storing vectors to pinecone..."):
        batch_chunks = chunks[i:i + batch_size] # Store text chunks of the RFPs by 100 (descriptions) 
        batch_metadata = metadata_list[i:i + batch_size] # Store the corresponding metadata in batches of 100
        embeddings = get_embeddings_batch(batch_chunks)  # Embedding the current batch of 100 text chunks 
        
        vectors = []
        # embedding : individual vector for the description of each RFP
        # metadata : individual dictionary for the corresponding metadata of the description
        for chunk_index, (embedding, metadata) in enumerate(zip(embeddings, batch_metadata), start=i):
            # Convert metadata values to acceptable formats
            for key, value in metadata.items():
                # Verify whether 'value' in metadata is a list and if all items within the list of value are not string, it returns true
                if isinstance(value, list) and not all(isinstance(item, str) for item in value):
                    metadata[key] = [str(item) for item in value]  # Convert all items in a list into string 
                elif not isinstance(value, (str, int, float, bool)): # If value is not 'str' , 'int' ,'float' and 'bool'
                    metadata[key] = str(value)  # Convert other types to string
            vector_id = f"{chunk_index}" # id는 반드시 str이어야 함
            vector_dict = format_to_vector_dict(vector_id, embedding, metadata)
            vectors.append(vector_dict)
        
        index.upsert(vectors=vectors, namespace=namespace) # Store vectors into Pinecone 

def collect_chunks_and_metadata_from_path(file_path):
    all_chunks = [] # Store all the descriptions 
    all_metadata = [] # Store the corresponding metadata (other information except for descriptions)
    dataset = load_jsonl(file_path) # Load the JSAONL file into list 
    for data in dataset:
        description = data["description"]
        metadata = {key: value for key, value in data.items() if key != "description"}
        all_chunks.append(description) # Only store 'description' 
        all_metadata.append(metadata) # Store all key: value pairs if key is not description 
    return all_chunks, all_metadata

if __name__ == "__main__":
    index = pc.Index(args.index) # Define index of Pincone which is the exact location for storing vectors 

    chunks, metadata = collect_chunks_and_metadata_from_path(args.jsonl_path) # Store description chunks and metadata for the corresponding descriptions 
    store_vector_to_pinecone(index, chunks, metadata, args.namespace) # Embedding the texts and store the results in Pinecone 
