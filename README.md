# EvenementsRAG - Historical Events RAG System

A progressive Retrieval-Augmented Generation (RAG) system for historical events, demonstrating the evolution from classical RAG to advanced Graph RAG approaches.

## Overview

EvenementsRAG is an educational and research project that builds **four progressively sophisticated RAG systems** to answer questions about World War II historical events. Each phase demonstrates different RAG capabilities and their strengths on various question types.

### The Four RAG Phases

1. **Phase 1: Classical Vanilla RAG**
   - Simple semantic search using embeddings
   - Best for: Factual questions, entity queries
   - Baseline implementation

2. **Phase 2: Temporal RAG**
   - Adds date filtering and temporal reasoning
   - Best for: Chronological questions, time-based queries
   - 30%+ improvement on temporal questions

3. **Phase 3: Hybrid RAG**
   - Combines semantic + keyword search (BM25) with reranking
   - Best for: Complex retrieval, comparisons, entity-heavy queries
   - 20%+ improvement on precision

4. **Phase 4: Graph RAG**
   - Knowledge graph with multi-hop reasoning
   - Best for: Causal chains, relationships, analytical questions
   - 50%+ improvement on multi-hop questions

## Project Structure

```
EvenementsRAG/
├── docs/
│   └── question_types_taxonomy.md    # Question categorization & examples
├── config/
│   ├── settings.py                   # Centralized configuration
│   └── periods/                      # Historical period configs
├── src/
│   ├── data_ingestion/              # Wikipedia fetching & parsing
│   ├── preprocessing/               # Text chunking & metadata
│   ├── embeddings/                  # Embedding generation
│   ├── vector_store/                # Qdrant interface
│   ├── graph_store/                 # Knowledge graph (Phase 4)
│   ├── rag/                         # RAG implementations
│   │   ├── phase1_vanilla/
│   │   ├── phase2_temporal/
│   │   ├── phase3_hybrid/
│   │   └── phase4_graph/
│   └── evaluation/                  # Metrics & benchmarking
├── data/                            # Data storage
├── notebooks/                       # Jupyter notebooks
├── scripts/                         # Utility scripts
└── tests/                          # Unit tests
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker (for Qdrant vector database)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EvenementsRAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

   Or using Poetry:
   ```bash
   poetry install
   ```

4. **Install spaCy language model**
   ```bash
   python -m spacy download en_core_web_lg
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

6. **Start Qdrant vector database**
   ```bash
   ./scripts/setup_qdrant.sh start
   ```

7. **Verify setup**
   ```bash
   python config/settings.py
   ```

## Configuration

All configuration is managed through environment variables in `.env` file:

### Required API Keys

```bash
# OpenRouter (Recommended - Free Mistral models available)
# Get your free key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key_here

# OR use other providers (optional):
# Anthropic (for Claude models)
ANTHROPIC_API_KEY=your_key_here

# OpenAI (for GPT models)
OPENAI_API_KEY=your_key_here
```

### Key Configuration Options

```bash
# LLM Provider (openrouter, anthropic, or openai)
LLM_PROVIDER=openrouter

# OpenRouter model (free Mistral model)
OPENROUTER_MODEL=mistralai/mistral-small-3.1-24b-instruct:free

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Qdrant Settings
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Retrieval Settings
DEFAULT_TOP_K=5
CHUNK_SIZE=512
```

See `.env.example` for all available options.

## Usage

### Managing Qdrant

```bash
# Start Qdrant
./scripts/setup_qdrant.sh start

# Check status
./scripts/setup_qdrant.sh status

# View logs
./scripts/setup_qdrant.sh logs

# Stop Qdrant
./scripts/setup_qdrant.sh stop
```

### Data Collection (Coming Soon)

```bash
# Download Wikipedia articles for WW2 period
python scripts/download_wikipedia_data.py --period ww2 --max-articles 100
```

### Running RAG Phases (Coming Soon)

```bash
# Process data pipeline
python scripts/process_data_pipeline.py

# Run Phase 1 (Vanilla RAG)
python -m src.rag.phase1_vanilla.run

# Run evaluation
python scripts/run_evaluation.py --phase all
```

## Development Roadmap

### ✅ Phase 0: Setup (Current)
- [x] Project structure
- [x] Question types taxonomy
- [x] Configuration management
- [x] Qdrant setup script
- [ ] Wikipedia data fetcher
- [ ] Initial dataset (50 WW2 articles)

### 🔄 Phase 1: Vanilla RAG (Weeks 3-4)
- [ ] Text chunking implementation
- [ ] Embedding generation
- [ ] Qdrant indexing
- [ ] Simple retrieval pipeline
- [ ] LLM generation
- [ ] Baseline evaluation

### 📋 Phase 2: Temporal RAG (Weeks 5-6)
- [ ] Date extraction from text
- [ ] Temporal metadata enrichment
- [ ] Date range filtering
- [ ] Chronological sorting
- [ ] Comparison with Phase 1

### 📋 Phase 3: Hybrid RAG (Weeks 7-8)
- [ ] BM25 keyword search
- [ ] Reciprocal Rank Fusion
- [ ] Cross-encoder reranking
- [ ] Entity extraction & boosting
- [ ] 3-way comparison

### 📋 Phase 4: Graph RAG (Weeks 9-12)
- [ ] Knowledge graph construction
- [ ] Relationship extraction
- [ ] Graph traversal & path finding
- [ ] Subgraph retrieval
- [ ] Full evaluation report

## Question Types

This project evaluates RAG systems on 6 major question categories:

1. **Factual**: "When did D-Day occur?"
2. **Temporal**: "What happened before Pearl Harbor?"
3. **Comparative**: "How did Stalingrad differ from Kursk?"
4. **Entity-Centric**: "What role did Churchill play?"
5. **Relationship**: "How did X influence Y?"
6. **Analytical**: "Summarize key events of 1944"

See [`docs/question_types_taxonomy.md`](docs/question_types_taxonomy.md) for comprehensive taxonomy with 175+ example questions.

## Evaluation Metrics

### Retrieval Metrics
- Recall@K (K=5,10,20)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### Generation Metrics
- ROUGE-L (lexical overlap)
- BERTScore (semantic similarity)
- Faithfulness (RAGAS)
- Answer Relevance (RAGAS)

### Specialized Metrics
- Temporal accuracy (for Phase 2)
- Path correctness (for Phase 4)
- Multi-hop success rate (for Phase 4)

## Technologies

### Core Stack
- **Python 3.10+**: Primary language
- **Qdrant**: Vector database for embeddings
- **sentence-transformers**: Embedding generation
- **LangChain**: LLM orchestration
- **OpenRouter (Mistral)**: LLM generation (free tier available)
- **Alternative LLMs**: Anthropic Claude / OpenAI GPT (optional)

### NLP & Processing
- **spaCy**: Named entity recognition
- **dateparser**: Date extraction
- **rank-bm25**: Keyword search

### Graph (Phase 4)
- **NetworkX**: Graph database (development)
- **Neo4j**: Graph database (optional, production)

### Evaluation
- **RAGAS**: RAG evaluation framework
- **BERTScore**: Semantic similarity
- **ROUGE**: Lexical similarity

## Data Source

Historical events data is sourced from **Wikipedia**, focusing on:
- **Period**: World War II (1939-1945)
- **Articles**: 50-500 articles
- **Categories**:
  - Battles of World War II
  - WW2 conferences & treaties
  - Military operations
  - Key historical figures

## Project Goals

1. **Educational**: Demonstrate evolution of RAG techniques
2. **Comparative**: Quantify improvements across RAG phases
3. **Practical**: Provide reusable RAG implementations
4. **Research**: Explore question-type specific RAG optimization

## Expected Results

Based on our taxonomy, we expect:
- **Phase 1**: 70-80% accuracy on factual questions
- **Phase 2**: 30%+ improvement on temporal questions
- **Phase 3**: 20%+ improvement on complex retrieval
- **Phase 4**: 50%+ improvement on multi-hop reasoning

## Contributing

This is currently a personal research project. Feedback and suggestions are welcome via issues.

## License

MIT License (to be added)

## Acknowledgments

- Wikipedia for historical data
- OpenRouter for free LLM access
- Mistral AI for open models
- Qdrant team for vector database
- LangChain community

## Contact

For questions or collaboration: [Your contact info]

---

**Status**: 🚧 Phase 0 Setup (In Progress)

**Last Updated**: 2025-12-30
