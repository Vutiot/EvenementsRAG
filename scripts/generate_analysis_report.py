#!/usr/bin/env python3
"""
Generate a detailed analysis report with sample questions and worst-performing cases.

This script:
1. Samples 10 random questions from the evaluation set
2. Identifies the 10 worst-ranking questions
3. Shows top 3 retrieved chunks for each question
4. Generates a markdown report with tables
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager
from src.vector_store.indexer import DocumentIndexer
from src.embeddings.embedding_generator import EmbeddingGenerator

logger = get_logger(__name__)


def load_questions(questions_file: Path) -> List[Dict]:
    """Load questions from JSON file."""
    with open(questions_file, 'r') as f:
        data = json.load(f)
    return data.get('questions', [])


def load_results(results_file: Path) -> Dict:
    """Load evaluation results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def retrieve_top_k(
    qdrant: QdrantManager,
    embedding_gen: EmbeddingGenerator,
    collection_name: str,
    question: str,
    k: int = 3
) -> List[Dict]:
    """Retrieve top-k chunks for a question."""
    # Generate embedding for question
    query_embedding = embedding_gen.generate_embeddings([question])[0]

    # Search
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )

    # Format results
    chunks = []
    for result in results:
        # Handle both dict and object formats
        if isinstance(result, dict):
            chunks.append({
                'score': result.get('score', 0.0),
                'text': result.get('payload', {}).get('content', ''),
                'article_title': result.get('payload', {}).get('article_title', ''),
                'chunk_id': result.get('payload', {}).get('chunk_id', 0),
            })
        else:
            chunks.append({
                'score': result.score,
                'text': result.payload.get('content', ''),
                'article_title': result.payload.get('article_title', ''),
                'chunk_id': result.payload.get('chunk_id', 0),
            })

    return chunks


def calculate_question_rank(question: Dict, chunks: List[Dict]) -> int:
    """Calculate the rank of the correct chunk (1-indexed, 0 if not found)."""
    source_chunk_id = question.get('source_chunk_id')
    source_article = question.get('source_article_id') or question.get('source_article_title')

    for idx, chunk in enumerate(chunks, 1):
        # Check if this is the right chunk
        if chunk.get('chunk_id') == source_chunk_id:
            return idx
        # Or at least from the right article
        if chunk.get('article_title') == source_article:
            # Found article but not exact chunk - consider this as found at position
            pass

    return 0  # Not found


def load_source_document(articles_dir: Path, article_title: str) -> str:
    """Load source document content by article title."""
    # Sanitize filename
    safe_title = article_title.replace("/", "_").replace("\\", "_")
    safe_title = safe_title.replace(":", "_").replace("*", "_")
    safe_title = safe_title.replace("?", "_").replace('"', "_")
    safe_title = safe_title.replace("<", "_").replace(">", "_")
    safe_title = safe_title.replace("|", "_")
    safe_title = safe_title[:200]

    filepath = articles_dir / f"{safe_title}.json"

    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('content', '')
        except:
            return "Error loading document"
    return "Document not found"


def generate_markdown_report(
    sample_questions: List[Tuple[Dict, List[Dict]]],
    worst_questions: List[Tuple[Dict, List[Dict], int]],
    output_file: Path,
    articles_dir: Path
):
    """Generate markdown report with tables."""

    with open(output_file, 'w') as f:
        f.write("# RAG Retrieval Analysis Report - 10,000 Article Dataset\n\n")
        f.write(f"**Generated:** {Path(output_file).name}\n\n")
        f.write("This report analyzes retrieval quality for the 10,000-article WW2 knowledge base.\n\n")

        # Part 1: Sample Questions
        f.write("---\n\n")
        f.write("## Part 1: Sample Questions (10 Random Samples)\n\n")
        f.write("This section shows 10 randomly sampled questions from the evaluation set to illustrate typical retrieval behavior.\n\n")

        for idx, (question, chunks) in enumerate(sample_questions, 1):
            f.write(f"### Sample {idx}\n\n")

            # Question metadata
            f.write(f"**Question Type:** {question.get('type', 'Unknown')}\n\n")
            f.write(f"**Source Article:** {question.get('source_article', 'Unknown')}\n\n")
            f.write(f"**Question:**\n> {question.get('question', '')}\n\n")

            # Load source document
            source_article = question.get('source_article', '')
            if source_article:
                doc_content = load_source_document(articles_dir, source_article)
                doc_preview = doc_content[:1500].replace('\n', ' ')
                f.write(f"**Source Document (first 1500 chars):**\n> {doc_preview}...\n\n")

            # Load source document preview
            source_article_title = question.get('source_article', '')
            doc_preview = ""
            if source_article_title:
                doc_content = load_source_document(articles_dir, source_article_title)
                doc_preview = doc_content[:1500].replace('\n', ' ').replace('|', '\\|')

            # Top 3 retrieved chunks
            f.write("**Top 3 Retrieved Chunks:**\n\n")
            f.write("| Rank | Score | Article | Chunk Text | Source Document Preview (first 1500 chars) |\n")
            f.write("|------|-------|---------|------------|---------------------------------------------|\n")

            for rank, chunk in enumerate(chunks, 1):
                score = chunk['score']
                article = chunk['article_title']
                text = chunk['text'][:1500].replace('\n', ' ').replace('|', '\\|')

                # Mark if this is the source article
                marker = " ✅" if article == source_article else ""

                f.write(f"| {rank} | {score:.3f} | {article}{marker} | {text} | {doc_preview} |\n")

            f.write("\n")

        # Part 2: Worst Performing Questions
        f.write("---\n\n")
        f.write("## Part 2: Top 10 Worst-Ranking Questions\n\n")
        f.write("This section identifies cases where retrieval struggled the most. ")
        f.write("These represent failure modes and opportunities for improvement.\n\n")

        for idx, (question, chunks, rank) in enumerate(worst_questions, 1):
            f.write(f"### Worst Case {idx} - Rank: {rank if rank > 0 else 'Not Found'}\n\n")

            # Question metadata
            f.write(f"**Question Type:** {question.get('type', 'Unknown')}\n\n")
            f.write(f"**Source Article:** {question.get('source_article', 'Unknown')}\n\n")
            f.write(f"**Expected Chunk ID:** {question.get('source_chunk_id', 'Unknown')}\n\n")
            f.write(f"**Question:**\n> {question.get('question', '')}\n\n")

            # Expected answer (if available)
            if 'expected_answer_hint' in question:
                f.write(f"**Expected Answer Hint:**\n> {question.get('expected_answer_hint', '')}\n\n")

            # Load source document
            source_article = question.get('source_article', '')
            if source_article:
                doc_content = load_source_document(articles_dir, source_article)
                doc_preview = doc_content[:1500].replace('\n', ' ')
                f.write(f"**Source Document (first 1500 chars):**\n> {doc_preview}...\n\n")

            # Load source document preview
            source_article_title = question.get('source_article', '')
            doc_preview = ""
            if source_article_title:
                doc_content = load_source_document(articles_dir, source_article_title)
                doc_preview = doc_content[:1500].replace('\n', ' ').replace('|', '\\|')

            # Top 3 retrieved chunks
            f.write("**Top 3 Retrieved Chunks:**\n\n")
            f.write("| Rank | Score | Article | Chunk Text | Source Document Preview (first 1500 chars) |\n")
            f.write("|------|-------|---------|------------|---------------------------------------------|\n")

            for rank_num, chunk in enumerate(chunks, 1):
                score = chunk['score']
                article = chunk['article_title']
                text = chunk['text'][:1500].replace('\n', ' ').replace('|', '\\|')

                # Mark if this is the source article
                marker = " ✅" if article == source_article else ""

                f.write(f"| {rank_num} | {score:.3f} | {article}{marker} | {text} | {doc_preview} |\n")

            f.write("\n")

            # Analysis
            f.write("**Analysis:**\n")
            if rank == 0:
                f.write("- ❌ **Source chunk not found in top-10**\n")
                f.write("- This indicates a semantic mismatch between question and source content\n")
            else:
                f.write(f"- ⚠️ **Source chunk found at rank {rank}**\n")
                f.write(f"- The correct chunk was retrieved but ranked below position 3\n")

            f.write("\n")

        # Summary
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("### Key Observations\n\n")
        f.write("**Sample Questions:**\n")
        f.write("- Demonstrates typical retrieval behavior across different question types\n")
        f.write("- Shows how semantic search finds relevant chunks even with 10,000 articles\n\n")

        f.write("**Worst-Performing Questions:**\n")
        f.write("- Identifies challenging cases where retrieval struggles\n")
        f.write("- Common failure modes: semantic mismatch, ambiguous queries, multi-hop reasoning\n")
        f.write("- Opportunities for improvement: hybrid search, query expansion, reranking\n\n")

        f.write("### Recommendations\n\n")
        f.write("1. **Hybrid Search**: Combine BM25 + semantic for better article-level recall\n")
        f.write("2. **Query Expansion**: Expand ambiguous queries with related terms\n")
        f.write("3. **Reranking**: Use cross-encoder to rerank top-10 results\n")
        f.write("4. **Contextual Chunking**: Improve chunk boundaries to preserve semantic coherence\n\n")

    print(f"✓ Report saved to: {output_file}")


def main():
    print("=" * 70)
    print("Generate Detailed Analysis Report")
    print("=" * 70)
    print()

    # Configuration
    collection_name = "ww2_events_10000"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_10000_articles_50q.json"
    results_file = project_root / "results" / "phase1_baseline_10000articles_50q.json"
    output_file = project_root / "results" / "RETRIEVAL_ANALYSIS_10K.md"

    # Load questions and results
    print("Loading questions and results...")
    questions = load_questions(questions_file)
    results = load_results(results_file)
    print(f"  Loaded {len(questions)} questions")
    print()

    # Initialize Qdrant and indexer
    print("Initializing Qdrant and indexing 10,000 articles...")
    qdrant = QdrantManager(use_memory=True)
    embedding_gen = EmbeddingGenerator()
    indexer = DocumentIndexer(qdrant_manager=qdrant)

    # Index articles
    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"
    stats = indexer.index_all_articles(
        collection_name=collection_name,
        articles_dir=articles_dir,
        recreate_collection=True,
    )
    print(f"  Indexed {stats['chunks_indexed']:,} chunks from {stats['articles_loaded']:,} articles")
    print()

    # Sample 10 random questions
    print("Sampling 10 random questions...")
    random.seed(42)  # Reproducible sampling
    sample_qs = random.sample(questions, min(10, len(questions)))

    sample_questions = []
    for q in sample_qs:
        chunks = retrieve_top_k(qdrant, embedding_gen, collection_name, q['question'], k=3)
        sample_questions.append((q, chunks))
    print(f"  Sampled {len(sample_questions)} questions")
    print()

    # Find worst-ranking questions
    print("Identifying worst-ranking questions...")
    all_questions_with_ranks = []

    for q in questions:
        # Retrieve top 10 to check ranking
        chunks = retrieve_top_k(qdrant, embedding_gen, collection_name, q['question'], k=10)
        rank = calculate_question_rank(q, chunks)
        all_questions_with_ranks.append((q, chunks[:3], rank))  # Store only top 3 for report

    # Sort by rank (0 = not found is worst, then higher ranks)
    all_questions_with_ranks.sort(key=lambda x: (0 if x[2] == 0 else x[2], -x[2]), reverse=True)
    worst_questions = all_questions_with_ranks[:10]

    print(f"  Found {len(worst_questions)} worst cases")
    print()

    # Generate report
    print("Generating markdown report...")
    generate_markdown_report(sample_questions, worst_questions, output_file, articles_dir)
    print()

    print("=" * 70)
    print("✓ Analysis Report Complete!")
    print("=" * 70)
    print()
    print(f"Report saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
