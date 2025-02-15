from utils.utils import load_jsonl, retrieve_top_k_similar_docs
from dotenv import load_dotenv
import os
import argparse
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

def analyze_similarities_by_set(args):
    client = OpenAI(api_key=args.openai_api_key)
    pc = Pinecone(api_key=args.pinecone_api_key)
    index = pc.Index(args.index_name)
    
    data = load_jsonl(args.data_path)
    skill_sets = load_jsonl(args.skill_sets_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_distributions = {}
    
    for idx, skill_set in enumerate(tqdm(skill_sets), start=1):
        scores = []
        skills = skill_set['text']
        
        try:
            retrieved_docs = retrieve_top_k_similar_docs(skills, index, args.namespace, k=len(data))
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
            stats_path = os.path.join(args.output_dir, f'skill_set_{idx}_stats.txt')
            
            with open(stats_path, 'w') as f:
                f.write(f"Skill Set {idx} Statistics:\n")
                f.write(f"Skills: {skills}\n\n")
                f.write(f"Mean: {df['similarity_score'].mean():.4f}\n")
                f.write(f"Median: {df['similarity_score'].median():.4f}\n")
                f.write(f"Standard Deviation: {df['similarity_score'].std():.4f}\n")
                f.write(f"Min: {df['similarity_score'].min():.4f}\n")
                f.write(f"Max: {df['similarity_score'].max():.4f}\n")
                
        except Exception as e:
            print(f"Error processing skill set {idx}: {str(e)}")
            continue
    
    # Create distribution plots
    plot_path_1 = os.path.join(args.output_dir, 'distributions_1_15.png')
    plot_path_2 = os.path.join(args.output_dir, 'distributions_16_30.png')
    
    create_multi_distribution_plot(all_distributions, 0, 15, plot_path_1)
    create_multi_distribution_plot(all_distributions, 15, 30, plot_path_2)

def main():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--data_path', type=str, default='datasets/data.jsonl',
                      help='Path to RFP data file')
    parser.add_argument('--skill_sets_path', type=str, default='datasets/test_skill_sets.jsonl',
                      help='Path to skill sets file')
    
    # Pinecone settings
    parser.add_argument('--index_name', type=str, default='rfp',
                      help='Name of Pinecone index')
    parser.add_argument('--namespace', type=str, default='openai-no-chunk',
                      help='Namespace in Pinecone index')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='../../results/score_distributions',
                      help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Load API keys from environment
    load_dotenv()
    args.openai_api_key = os.environ.get("OPENAI_API_KEY")
    args.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    analyze_similarities_by_set(args)

if __name__ == "__main__":
    main()