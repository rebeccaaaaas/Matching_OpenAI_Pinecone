from utils.utils import load_jsonl, retrieve_top_k_similar_docs
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
import json
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RFP Matching Script')
    
    # Pinecone parameters
    parser.add_argument('--namespace', type=str, default='openai-no-chunk',
                      help='Namespace within Pinecone index')
    parser.add_argument('--index_name', type=str, default='rfp',
                      help='Name of the Pinecone index')
    
    # Input file paths
    parser.add_argument('--data_path', type=str, default='../../datasets/utility_rfps.jsonl',
                      help='Path to the RFP data file')
    parser.add_argument('--skill_sets_path', type=str, default='../../datasets/test_skill_sets.jsonl',
                      help='Path to the skill sets file')
    
    # Output file paths
    parser.add_argument('--output_matched_docs', type=str, default='../../results/utest_matched_docs.txt',
                      help='Path to save matched documents')
    parser.add_argument('--output_match_scores', type=str, default='../../results/utest_matchescores.txt',
                      help='Path to save matching scores')
    
    # Additional parameters
    parser.add_argument('--top_k', type=int, default=3,
                      help='Number of top matches to retrieve')
    
    return parser.parse_args()

def ensure_directory(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # API Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing required API keys in environment variables")
    
    # Initialize OpenAI client for embedding generation
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize Pinecone client for vector search
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(args.index_name)
    
    try:
        # Load data and skill sets
        data = load_jsonl(args.data_path)
        skill_sets = load_jsonl(args.skill_sets_path)
        
        # Ensure output directories exist
        ensure_directory(args.output_matched_docs)
        ensure_directory(args.output_match_scores)
        
        # Process each skill set and find matching RFPs
        for idx, skill_set in enumerate(tqdm(skill_sets), start=1):
            # Get skills text
            skills = skill_set['text']
            
            # Retrieve top k similar documents using vector similarity search
            retrieved_docs = retrieve_top_k_similar_docs(
                skills, 
                index, 
                args.namespace, 
                k=args.top_k
            )
            
            # Save detailed matching results
            with open(args.output_matched_docs, 'a') as f:
                f.write(f"{idx}. Skill Set: {skills}\nTop {args.top_k} Matched Documents:\n")
                
                for i, retrieved_doc in enumerate(retrieved_docs, start=1):
                    similarity_score = retrieved_doc['score']
                    posting_id = data[int(retrieved_doc['id'])]['postingId']
                    description = data[int(retrieved_doc['id'])]['description']
                    
                    f.write(f"{idx}-{i}. Posting ID: {posting_id}\n")
                    f.write(f"Similarity: {similarity_score:.4f}\n")
                    f.write(f"Description: {description}\n\n")
                    
                f.write("-" * 100 + "\n\n")
            
            # Save summary of matching scores
            with open(args.output_match_scores, 'a') as f:
                f.write(f"{idx}. Skill Set: {skills}\nTop {args.top_k} Matched Documents:\n")
                
                for i, retrieved_doc in enumerate(retrieved_docs, start=1):
                    similarity_score = retrieved_doc['score']
                    posting_id = data[int(retrieved_doc['id'])]['postingId']
                    
                    f.write(f"{idx}-{i}. Posting ID: {posting_id}\n")
                    f.write(f"Similarity: {similarity_score:.4f}\n\n")
                    
                f.write("-" * 100 + "\n\n")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()