"""
Text chunking for historical event documents.

Implements semantic chunking that:
- Preserves sentence and paragraph boundaries
- Uses token-based sizing
- Adds overlap between chunks
- Preserves metadata

Usage:
    from src.preprocessing.text_chunker import TextChunker

    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(article_data)
"""

import re
from typing import Dict, List, Optional

import tiktoken

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Chunks text documents for RAG processing."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_chunks_per_doc: Optional[int] = None,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            max_chunks_per_doc: Maximum chunks per document (None = unlimited)
            encoding_name: Tokenizer encoding name (cl100k_base for GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc or settings.MAX_CHUNKS_PER_DOC

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}: {e}")
            logger.info("Falling back to simple word-based tokenization")
            self.encoding = None

        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, max_chunks={self.max_chunks_per_doc}"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: estimate ~0.75 tokens per word
            return int(len(text.split()) * 0.75)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting - handles most cases
        # Matches periods, exclamation marks, question marks followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)

        # Clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[str]:
        """
        Chunk text into token-sized pieces.

        Args:
            text: Input text
            preserve_structure: Try to preserve sentence/paragraph boundaries

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Check if text fits in one chunk
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.chunk_size:
            return [text]

        if preserve_structure:
            return self._chunk_with_structure(text)
        else:
            return self._chunk_simple(text)

    def _chunk_simple(self, text: str) -> List[str]:
        """
        Simple chunking without preserving structure.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        if not self.encoding:
            # Word-based fallback
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0

            for word in words:
                word_tokens = 1  # Approximate
                if current_size + word_tokens > self.chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Add overlap
                        overlap_words = int(self.chunk_overlap * 1.33)  # tokens to words
                        current_chunk = current_chunk[-overlap_words:]
                        current_size = len(current_chunk)
                current_chunk.append(word)
                current_size += word_tokens

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        # Token-based chunking
        tokens = self.encoding.encode(text)
        chunks = []

        i = 0
        while i < len(tokens):
            # Get chunk
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move forward with overlap
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def _chunk_with_structure(self, text: str) -> List[str]:
        """
        Chunk text while preserving paragraph and sentence boundaries.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)

            # If paragraph is too large, split by sentences
            if paragraph_tokens > self.chunk_size:
                # First, save current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph into sentences
                sentences = self.split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)

                    # If single sentence is too large, use simple chunking
                    if sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = []
                            current_tokens = 0

                        # Split the sentence
                        sentence_chunks = self._chunk_simple(sentence)
                        chunks.extend(sentence_chunks)

                    elif current_tokens + sentence_tokens > self.chunk_size:
                        # Start new chunk
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens

                    else:
                        # Add to current chunk
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens

            elif current_tokens + paragraph_tokens > self.chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens

            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        # Don't forget last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def chunk_document(
        self,
        article_data: Dict,
        content_field: str = "content",
    ) -> List[Dict]:
        """
        Chunk a Wikipedia article document with metadata.

        Args:
            article_data: Article data dictionary
            content_field: Field containing text content

        Returns:
            List of chunk dictionaries with metadata
        """
        content = article_data.get(content_field, "")

        if not content:
            logger.warning(f"No content found in field '{content_field}'")
            return []

        # Chunk the text
        text_chunks = self.chunk_text(content, preserve_structure=True)

        # Limit number of chunks
        if len(text_chunks) > self.max_chunks_per_doc:
            logger.warning(
                f"Document has {len(text_chunks)} chunks, limiting to {self.max_chunks_per_doc}"
            )
            text_chunks = text_chunks[:self.max_chunks_per_doc]

        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                # Content
                "content": chunk_text,
                "chunk_index": i,
                "total_chunks": len(text_chunks),

                # Source metadata
                "title": article_data.get("title", ""),
                "source_url": article_data.get("url", ""),
                "article_title": article_data.get("title", ""),

                # Article metadata (subset)
                "categories": article_data.get("categories", [])[:10],  # Limit
                "pageid": article_data.get("pageid"),

                # Chunk stats
                "token_count": self.count_tokens(chunk_text),
                "char_count": len(chunk_text),
            }

            chunks.append(chunk)

        logger.info(
            f"Chunked document '{article_data.get('title', 'unknown')}' into {len(chunks)} chunks",
            extra={"article": article_data.get("title"), "chunks": len(chunks)},
        )

        return chunks

    def chunk_documents(self, articles: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            articles: List of article dictionaries

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for article in articles:
            chunks = self.chunk_document(article)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(articles)} documents into {len(all_chunks)} total chunks",
            extra={"documents": len(articles), "chunks": len(all_chunks)},
        )

        return all_chunks

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "avg_chars": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }

        token_counts = [c.get("token_count", 0) for c in chunks]
        char_counts = [c.get("char_count", 0) for c in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
        }

        return stats


if __name__ == "__main__":
    # Test the chunker
    import json

    print("Testing TextChunker...")

    # Load a test article
    test_file = "data/raw/wikipedia_articles/Normandy_landings.json"

    try:
        with open(test_file, "r") as f:
            article = json.load(f)

        print(f"\nLoaded article: {article['title']}")
        print(f"Content length: {len(article['content'])} chars")

        # Initialize chunker
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)

        # Chunk the document
        chunks = chunker.chunk_document(article)

        print(f"\nCreated {len(chunks)} chunks")

        # Show statistics
        stats = chunker.get_chunk_statistics(chunks)
        print("\nChunk Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")

        # Show first chunk
        print("\nFirst chunk:")
        print(f"  Index: {chunks[0]['chunk_index']}/{chunks[0]['total_chunks']}")
        print(f"  Tokens: {chunks[0]['token_count']}")
        print(f"  Content preview: {chunks[0]['content'][:200]}...")

    except FileNotFoundError:
        print(f"\nTest file not found: {test_file}")
        print("Run the Wikipedia downloader first:")
        print("  python scripts/download_wikipedia_data.py --titles 'D-Day'")
