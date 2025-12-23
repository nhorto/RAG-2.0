#!/usr/bin/env python3
"""Ingest documents into Qdrant vector database."""

import argparse
import sys
from pathlib import Path
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_engine import ChunkingEngine
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.embedding_generator import HybridEmbeddingGenerator
from src.database.qdrant_client import QdrantManager
from src.utils.config_loader import get_config


def main():
    """Main ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default from config)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate collection (delete existing data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in batch",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="File pattern to match (default: *.txt, *.srt, *.vtt)",
    )

    args = parser.parse_args()

    # Initialize components
    print("Initializing RAG components...")
    config = get_config()

    loader = DocumentLoader()
    chunker = ChunkingEngine()
    metadata_extractor = MetadataExtractor()
    embedder = HybridEmbeddingGenerator()
    qdrant = QdrantManager(collection_name=args.collection)

    # Create or recreate collection (with sparse vector support)
    print(f"\nCollection: {qdrant.collection_name}")
    if args.recreate_collection or not qdrant.collection_exists():
        print("Creating Qdrant collection with hybrid (dense + sparse) vectors...")
        qdrant.create_collection(
            enable_sparse=True,
            recreate=args.recreate_collection
        )
    else:
        print("Using existing collection")

    # Load documents
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    print(f"\nLoading documents from: {source_dir}")
    patterns = [args.file_pattern] if args.file_pattern else None
    documents = loader.load_directory(source_dir, patterns=patterns)

    if not documents:
        print("No documents found to ingest")
        return 1

    print(f"Found {len(documents)} documents")

    # Process documents
    total_chunks = 0

    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing: {doc.filename}")

        # Chunk document
        doc_type = chunker.detect_document_type(doc.content, doc.filename)
        chunks = chunker.chunk_document(doc.content, doc.document_id, doc_type)

        print(f"  Created {len(chunks)} chunks")

        # Extract file metadata
        file_metadata = metadata_extractor.extract_file_metadata(
            doc.filename, doc.content
        )
        file_metadata["filename"] = doc.filename
        file_metadata["document_type"] = doc_type

        # Generate hybrid embeddings (dense + sparse)
        print(f"  Generating hybrid embeddings...")
        chunk_texts = [chunk.text for chunk in chunks]
        hybrid_embeddings = embedder.generate_embeddings(
            chunk_texts, show_progress=False
        )

        # Prepare chunks with metadata
        chunks_with_metadata = []
        for chunk, embedding in zip(chunks, hybrid_embeddings):
            # Create payload
            payload = metadata_extractor.create_chunk_payload(
                chunk_text=chunk.text,
                document_metadata=file_metadata,
                chunk_metadata={
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "char_count": chunk.char_count,
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                },
            )

            chunks_with_metadata.append({
                "chunk_id": chunk.chunk_id,
                **payload,
            })

        # Upsert to Qdrant with hybrid embeddings
        print(f"  Upserting to Qdrant with hybrid vectors...")
        qdrant.upsert_hybrid(
            chunks=chunks_with_metadata,
            hybrid_embeddings=hybrid_embeddings,
            batch_size=100
        )

        total_chunks += len(chunks)

    # Summary
    print(f"\n{'=' * 60}")
    print("Ingestion Complete!")
    print(f"{'=' * 60}")
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Collection: {qdrant.collection_name}")

    # Verify
    info = qdrant.get_collection_info()
    print(f"Points in collection: {info.get('points_count', 'Unknown')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
