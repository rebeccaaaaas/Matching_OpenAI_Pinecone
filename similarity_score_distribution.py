from utils import load_jsonl, retrieve_top_k_similar_docs
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_multi_distribution_plot(all_data, start_idx, end_idx, filename):
    fig, axes = plt.subplots(3, 5, figsize=(25, 15)) 
    axes = axes.ravel()  
    
    for idx, (skill_idx, data) in enumerate(list(all_data.items())[start_idx:end_idx]):
        ax = axes[idx]
        df = pd.DataFrame(data['scores'])
        
        sns.histplot(data=df, x='similarity_score', bins=30, ax=ax)
        ax.set_title(f'Skill Set {skill_idx}', fontsize=10)
        ax.set_xlabel('Similarity Score', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        
        stats_text = f'Mean: {df["similarity_score"].mean():.4f}\n'
        stats_text += f'Median: {df["similarity_score"].median():.4f}\n'
        stats_text += f'Std: {df["similarity_score"].std():.4f}'
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_similarities_by_set():
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("rfp")
    namespace = "openai-no-chunk"
    
    data = load_jsonl("datasets/data.jsonl")
    skill_sets = load_jsonl("datasets/test_skill_sets.jsonl")
    
    os.makedirs('score_distributions', exist_ok=True)
    
    all_distributions = {}
    
    for idx, skill_set in enumerate(tqdm(skill_sets), start=1):
        scores = []
        skills = skill_set['text']
        
        try:
            retrieved_docs = retrieve_top_k_similar_docs(skills, index, namespace, k=len(data))
            for doc in retrieved_docs:
                scores.append({
                    'rfp_id': data[int(doc['id'])]['postingId'],
                    'similarity_score': doc['score']
                })
            
            all_distributions[idx] = {
                'skills': skills,
                'scores': scores
            }
            
            df = pd.DataFrame(scores)
            with open(f'score_distributions/skill_set_{idx}_stats.txt', 'w') as f:
                f.write(f"Skill Set {idx} Statistics:\n")
                f.write(f"Skills: {skills}\n\n")
                f.write(f"Mean: {df['similarity_score'].mean():.4f}\n")
                f.write(f"Median: {df['similarity_score'].median():.4f}\n")
                f.write(f"Standard Deviation: {df['similarity_score'].std():.4f}\n")
                f.write(f"Min: {df['similarity_score'].min():.4f}\n")
                f.write(f"Max: {df['similarity_score'].max():.4f}\n")
                
        except Exception as e:
            print(f"处理skill set {idx}时出错: {str(e)}")
            continue
    
    create_multi_distribution_plot(all_distributions, 0, 15, 'score_distributions/distributions_1_15.png')
    create_multi_distribution_plot(all_distributions, 15, 30, 'score_distributions/distributions_16_30.png')

if __name__ == "__main__":
    analyze_similarities_by_set()