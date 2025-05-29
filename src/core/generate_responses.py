#!/usr/bin/env python3
"""
RFP Response Generator with Integrated Matching
Performs similarity matching, scores RFPs, and generates responses for top performers
"""

import os
import json
import pandas as pd
import math
import time
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Response generation template
RESPONSE_TEMPLATE = """
Given the Summary of Amplytics, write an RFP response that describes how Amplytics can address the requirements. 
The response should follow the predefined template.

Summary of Amplytics:
{summary}

Predefined Template:
'''
Subject: Response to [RFP Title / Reference Number]

Dear [Recipient Name],

We are pleased to submit our response to [RFP Title / Reference Number]. Amplytics, a consulting firm specializing in data-driven decision-making, offers innovative solutions tailored to the utilities industry. With over [X years] of experience, we have successfully helped utilities optimize their operations and achieve sustainable growth through advanced data analytics, machine learning, and digital transformation.

Introduction & Company Overview:
Amplytics transforms utilities businesses into data-powered organizations by integrating robust analytics, strategic advisory, and cutting-edge technology. Our core capabilities include: 
[RFP-Specific Capabilities – Adjust as needed]

Understanding of RFP Scope and Requirements:
We understand that [Agency/Organization Name] seeks [Brief Summary of RFP Requirements]. Our team is well-equipped to address the challenges presented, including:
• [Key Requirement 1 – From RFP]
• [Key Requirement 2 – From RFP]
• [Key Requirement 3 – From RFP]

Proposed Approach & Methodology:
Our phased approach ensures efficient implementation and alignment with your organizational goals:
[Customize Phases Based on RFP Requirements]

Amplytics' Value Proposition:
[Reiterate Skills & Expertise Relevant to RFP]

Relevant Experience & Past Performance:
[Insert Relevant Case Studies Related to RFP]

Deliverables Summary:
[Summarize Key Deliverables Aligned with RFP]

Team & Capabilities:
[Insert Team Member Profiles – Adjust as Needed]

Conclusion:
We are confident that Amplytics is the ideal partner for [Agency/Organization Name]'s [RFP Objective]. Our expertise and tailored solutions will ensure your objectives are achieved efficiently and effectively.

Please feel free to reach out with any questions or further information.

Best regards,
[Your Full Name]
[Your Title]
Amplytics
[Contact Information]
'''

Only generate the RFP response.
"""


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Generate embedding using OpenAI API"""
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def retrieve_top_k_similar_docs(query: str, index, namespace: str, k: int = 100) -> List[Dict]:
    """Retrieve top-k similar documents from Pinecone"""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Extract matches with scores
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score']
            })
        
        return matches
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []


def calculate_similarity_scores(skill_sets: List[Dict], data: List[Dict], 
                              index, namespace: str) -> Dict[str, float]:
    """Calculate similarity scores for all RFPs against skill sets"""
    print("Calculating similarity scores...")
    
    # Create mapping from index to postingId
    posting_id_map = {}
    for idx, item in enumerate(data):
        posting_id_map[str(idx)] = item.get('postingId', '')
    
    # Store the highest similarity score for each RFP
    rfp_scores = {}
    
    for skill_set in tqdm(skill_sets, desc="Processing skill sets"):
        skills = skill_set.get('text', '')
        if not skills:
            continue
            
        # Get similar documents for this skill set
        similar_docs = retrieve_top_k_similar_docs(skills, index, namespace, k=len(data))
        
        # Update scores with the highest similarity for each RFP
        for doc in similar_docs:
            doc_id = str(doc['id'])
            score = doc['score']
            posting_id = posting_id_map.get(doc_id, '')
            
            if posting_id and (posting_id not in rfp_scores or rfp_scores[posting_id] < score):
                rfp_scores[posting_id] = score
    
    return rfp_scores


def generate_response(description: str, max_retries: int = 3) -> str:
    """Generate response using OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            prompt = RESPONSE_TEMPLATE.format(summary=description)
            
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error generating response: {str(e)}"


def create_rfp_dataframe(data: List[Dict], scores: Dict[str, float]) -> pd.DataFrame:
    """Create DataFrame with RFP data and scores"""
    rfp_records = []
    
    for item in data:
        posting_id = item.get('postingId', '')
        score = scores.get(posting_id, 0.0)
        
        record = {
            'score': score,
            'contracting_office_address': item.get('contracting_office_address', ''),
            'created_at': item.get('created_at', ''),
            'department': item.get('department', ''),
            'description': item.get('description', ''),
            'downloadUrl': item.get('downloadUrl', ''),
            'generalInfos': item.get('generalInfos', ''),
            'hostingUrls': item.get('hostingUrls', ''),
            'id': item.get('id', ''),
            'postingId': posting_id,
            'primary_poc': item.get('primary_poc', ''),
            'secondary_poc': item.get('secondary_poc', ''),
            'status': item.get('status', ''),
            'title': item.get('title', ''),
            'url': item.get('url', ''),
            'response': ''  # Will be filled for top performers
        }
        rfp_records.append(record)
    
    return pd.DataFrame(rfp_records)


def main():
    parser = argparse.ArgumentParser(description='Generate responses for top-scoring RFPs')
    parser.add_argument('--data_path', default='../../datasets/data.jsonl',
                        help='Path to RFP data file')
    parser.add_argument('--skill_sets_path', default='../../datasets/test_skill_sets.jsonl',
                        help='Path to skill sets file')
    parser.add_argument('--index_name', default='rfp',
                        help='Name of the Pinecone index')
    parser.add_argument('--namespace', default='openai-no-chunk',
                        help='Namespace within the Pinecone index')
    parser.add_argument('--output_csv', default='../../results/rfp_responses.csv',
                        help='Path to output CSV file')
    parser.add_argument('--top_percentage', type=float, default=0.1,
                        help='Percentage of top RFPs to generate responses for (default: 0.1 for 10%)')
    parser.add_argument('--delay', type=float, default=3.0,
                        help='Delay between API calls in seconds (default: 3.0)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading RFP data from {args.data_path}...")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} RFP records")
    
    print(f"Loading skill sets from {args.skill_sets_path}...")
    skill_sets = load_jsonl(args.skill_sets_path)
    print(f"Loaded {len(skill_sets)} skill sets")
    
    # Connect to Pinecone
    print(f"Connecting to Pinecone index: {args.index_name}")
    index = pc.Index(args.index_name)
    
    # Calculate similarity scores
    scores = calculate_similarity_scores(skill_sets, data, index, args.namespace)
    print(f"Calculated scores for {len(scores)} RFPs")
    
    # Create DataFrame and sort by scores
    print("Creating DataFrame and sorting by scores...")
    df = create_rfp_dataframe(data, scores)
    df = df.sort_values(by='score', ascending=False)
    
    # Calculate number of RFPs for response generation
    top_count = math.ceil(len(df) * args.top_percentage)
    print(f"Generating responses for top {top_count} RFPs ({args.top_percentage*100:.1f}%)")
    
    # Generate responses for top percentage
    for idx in tqdm(range(top_count), desc="Generating responses"):
        description = df.iloc[idx]['description']
        posting_id = df.iloc[idx]['postingId']
        score = df.iloc[idx]['score']
        
        print(f"Generating response for RFP {posting_id} (score: {score:.4f})...")
        
        if description and isinstance(description, str) and description.strip():
            response = generate_response(description)
            df.at[df.index[idx], 'response'] = response
        else:
            df.at[df.index[idx], 'response'] = "No valid description available for response generation."
        
        # Add delay between API calls to avoid rate limits
        if idx < top_count - 1:
            time.sleep(args.delay)
    
    # Save to CSV
    print(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nCompleted!")
    print(f"Total RFPs processed: {len(df)}")
    print(f"Responses generated: {top_count}")
    print(f"Results saved to: {args.output_csv}")
    
    # Print summary statistics
    print(f"\nScore Statistics:")
    print(f"Highest score: {df['score'].max():.4f}")
    print(f"Lowest score: {df['score'].min():.4f}")
    print(f"Mean score: {df['score'].mean():.4f}")
    print(f"Scores > 0: {(df['score'] > 0).sum()}")
    
    # Show top 5 scoring RFPs
    print(f"\nTop 5 RFPs by score:")
    top_5 = df.head(5)[['postingId', 'score', 'title']].copy()
    top_5['title'] = top_5['title'].str[:50] + '...'  # Truncate for display
    print(top_5.to_string(index=False))


if __name__ == "__main__":
    main()