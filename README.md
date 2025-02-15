# Matching_OpenAI_Pinecone
# RFP Matching Algorithm

This project implements an intelligent RFP (Request for Proposal) matching system that uses embeddings and similarity scoring to find relevant RFPs based on skill sets and requirements.

## Setup

### Prerequisites

1. OpenAI API Key
- Obtain an OpenAI API key
- Create a `.env` file in the project root
- Add your API key as: `OPENAI_API_KEY=your_key_here`

2. Python Environment
```bash
conda create -n rfp python=3.12
conda activate rfp
pip install -r requirements.txt
```

## Core Components

### 1. Data Extraction (`extract_data.py`)
- Extracts data from Supabase database
- Consolidates information into a single JSONL file
```bash
python extract_data.py
```

### 2. Vector Database Storage (`store_pinecone.py`)
- Stores data in Pinecone vector database for efficient similarity search
```bash
python store_pinecone.py \
--index rfp \
--namespace openai-no-chunk \
--jsonl_path datasets/data.jsonl
```

### 3. Similarity Scoring (`compare_sim_score.py`)
- Implements cosine similarity calculation
- Uses OpenAI's text-embedding-3-large model
- Handles text truncation for API limits
- Key functions:
  - `get_embedding()`: Generates embeddings using OpenAI API
  - `truncate_text_tokens()`: Ensures text fits within token limits
  - `cosine_similarity()`: Calculates similarity between embeddings

### 4. Similarity Analysis (`similarity_score_distribution.py`)
- Analyzes similarity score distributions across different skill sets
- Creates visualization plots using matplotlib and seaborn
- Generates statistical reports including:
  - Mean similarity scores
  - Median values
  - Standard deviations
  - Score distributions

### 5. Utility Industry Filter (`utility_data.py`)
- Filters RFPs specific to the utility industry
- Uses OpenAI's GPT-3.5-turbo for classification
- Interfaces with Supabase for data storage

## Usage

### Core Workflow

1. Extract Data:
```bash
python extract_data.py
```
Extracts data from Supabase database and consolidates it into a JSONL file.

2. (Optional) Filter Utility Industry RFPs:
```bash
python utility_data.py
```
Optionally filter RFPs specific to the utility industry using OpenAI's GPT model.

3. Store in Vector Database:
```bash
python store_pinecone.py --index rfp --namespace openai-no-chunk --jsonl_path datasets/data.jsonl
```
Uploads the processed data to Pinecone for vector search.

Parameters:
- `--index`: Name of the Pinecone index (default: "rfp")
- `--namespace`: Namespace within the index (default: "openai-no-chunk")
- `--jsonl_path`: Path to the input JSONL file (default: "datasets/data.jsonl")

4. Run Inference:
```bash
python inference.py
```
Performs similarity matching to find relevant RFPs based on skill sets.

### Analysis Tools (Optional)

After running the core workflow, you can use these tools for analysis:

5. Analyze Score Distributions:
```bash
python similarity_score_distribution.py
```
Generates visualizations and statistics showing the distribution of similarity scores for each skill set across all RFPs. This helps understand the overall matching patterns and identify potential thresholds.

6. Compare Selected RFPs:
```bash
python compare_sim_score.py
```
Calculates similarity scores for manually selected RFPs that are known to match the requirements. This helps validate the matching algorithm and provides benchmark scores for comparison.

## Project Structure

```
project-root/
├── .env                              # Environment variables
├── compare_sim_score.py             # Similarity calculation
├── similarity_score_distribution.py  # Analysis visualization
├── utility_data.py                  # Industry filtering
├── extract_data.py                  # Data extraction
├── store_pinecone.py               # Vector DB storage
├── inference.py                    # Matching inference
└── datasets/
    ├── data.jsonl                  # Processed data
    └── test_skill_sets.jsonl       # Test cases
```

## Configuration

Required environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Technical Details

### Embedding Generation
- Model: text-embedding-3-large
- Max tokens: 8191
- Embedding dimension: 1024

### Similarity Scoring
- Method: Cosine similarity
- Score range: 0 to 1
- Higher scores indicate better matches

### Data Processing
- Text truncation for API limits
- JSON Lines (JSONL) format for data storage
- Vectorized storage in Pinecone for efficient retrieval

## Dependencies

- OpenAI
- Pinecone
- Supabase
- pandas
- matplotlib
- seaborn
- tiktoken
- python-dotenv
- tqdm

## Notes

- Ensure sufficient API credits for OpenAI services
- Monitor Pinecone database usage and limits
- Regular backup of Supabase data recommended
- Check token usage when processing large RFPs
