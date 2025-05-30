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
python extract_data.py --output_path ../../datasets/data.jsonl
```

### 2. Vector Database Storage (`store_pinecone.py`)
- Stores data in Pinecone vector database for efficient similarity search
```bash
python store_pinecone.py \
--index rfp \
--namespace openai-no-chunk \
--jsonl_path ../../datasets/data.jsonl
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

### 5. Response Generation (`generate_responses.py`)
- Performs integrated similarity matching using Pinecone vector database
- Calculates similarity scores for all RFPs against skill sets
- Sorts RFPs by similarity scores in descending order
- Generates AI responses for top 10% highest-scoring RFPs using OpenAI's GPT model
- Creates comprehensive CSV output with all RFP data, scores, and responses
- Key functions:
  - `calculate_similarity_scores()`: Performs similarity matching and scoring
  - `generate_response()`: Creates tailored responses using predefined templates
  - `create_rfp_dataframe()`: Consolidates all RFP data with scores

### 6. Utility Industry Filter (`utility_data.py`)
- Filters RFPs specific to the utility industry
- Uses OpenAI's GPT-3.5-turbo for classification
- Interfaces with Supabase for data storage

## Usage

### Core Workflow

1. Extract Data:
```bash
python extract_data.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --output_path | ../../datasets/data.jsonl | Path to save the extracted data |

Extracts data from Supabase database and consolidates it into a JSONL file.

2. (Optional) Filter Utility Industry RFPs:
```bash
python utility_data.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --output_path | datasets/utility_rfps.jsonl | Path to save filtered utility RFPs |

Optionally filter RFPs specific to the utility industry using OpenAI's GPT model.

3. Store in Vector Database:
```bash
python store_pinecone.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --index | rfp | Name of the Pinecone index |
| --namespace | openai-no-chunk | Namespace within the index |
| --jsonl_path | datasets/data.jsonl | Path to input JSONL file |

Uploads the processed data to Pinecone for vector search.

4. Run Inference:
```bash
python inference.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --namespace | openai-no-chunk | Namespace within Pinecone index |
| --index_name | rfp | Name of the Pinecone index |
| --data_path | ../../datasets/utility_rfps.jsonl | Path to the RFP data file |
| --skill_sets_path | ../../datasets/test_skill_sets.jsonl | Path to the skill sets file |
| --output_matched_docs | ../../results/utest_matched_docs.txt | Path to save matched documents |
| --output_match_scores | ../../results/utest_matchescores.txt | Path to save matching scores |
| --top_k | 3 | Number of top matches to retrieve |

The script will:
- Load RFP data and skill sets from specified paths
- Perform similarity matching using Pinecone
- Generate two output files:
  1. Detailed matches with descriptions
  2. Summary of matching scores

5. Generate Responses for Top RFPs:
```bash
python generate_responses.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --data_path | ../../datasets/data.jsonl | Path to RFP data file |
| --skill_sets_path | ../../datasets/test_skill_sets.jsonl | Path to skill sets file |
| --index_name | rfp | Name of the Pinecone index |
| --namespace | openai-no-chunk | Namespace within the Pinecone index |
| --output_csv | ../../results/rfp_responses.csv | Path to output CSV file |
| --top_percentage | 0.1 | Percentage of top RFPs to generate responses for |
| --delay | 3.0 | Delay between API calls in seconds |

Performs similarity matching against skill sets, calculates scores for all RFPs, sorts them by relevance, and generates AI-powered responses for the highest-scoring opportunities. The output CSV contains all RFP data with similarity scores and responses for top performers.

### Analysis Tools (Optional)

After running the core workflow, you can use these tools for analysis:

6. Analyze Score Distributions:
```bash
python analyze_similarity.py [arguments]
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| --data_path | datasets/data.jsonl | Path to RFP data file |
| --skill_sets_path | datasets/test_skill_sets.jsonl | Path to skill sets file |
| --index_name | rfp | Name of Pinecone index |
| --namespace | openai-no-chunk | Namespace in Pinecone index |
| --output_dir | score_distributions | Directory to save output files |

Generates visualizations and statistics showing the distribution of similarity scores for each skill set across all RFPs. This helps understand the overall matching patterns and identify potential thresholds.

7. Compare Selected RFPs:
```bash
python compare_sim_score.py
```
Calculates similarity scores for manually selected RFPs that are known to match the requirements. This helps validate the matching algorithm and provides benchmark scores for comparison.

## Project Structure

```
project-root/
├── .env                           # Environment variables
├── src/
│   ├── core/                     # Core workflow scripts
│   │   ├── extract_data.py      # Data extraction from Supabase
│   │   ├── store_pinecone.py    # Vector DB storage
│   │   ├── inference.py         # Main matching logic
│   │   ├── generate_responses.py # Response generation and CSV export
│   │   └── utility_data.py      # Optional utility industry filter
│   │
│   ├── analysis/                # Analysis tools
│   │   ├── compare_sim_score.py        # Compare selected RFPs
│   │   └── similarity_score_distribution.py  # Score distribution analysis
│   │
│   └── utils/                   # Shared utilities
│       └── utils.py        
│
├── datasets/                    # Data storage
│   └── test_skill_sets.jsonl   # Test cases
│
└── results/                    # Analysis outputs

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

### Response Generation
- Model: GPT-3.5-turbo for cost efficiency
- Max tokens: 2000 per response
- Temperature: 0.7 for balanced creativity and consistency
- Includes retry logic and rate limiting

### Data Processing
- Text truncation for API limits
- JSON Lines (JSONL) format for data storage
- Vectorized storage in Pinecone for efficient retrieval
- CSV output for dashboard integration and analysis

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
- The response generation feature uses GPT-3.5-turbo for cost optimization
- Consider adjusting the delay parameter if encountering rate limits
- CSV output includes all RFP fields plus similarity scores and responses for comprehensive analysis