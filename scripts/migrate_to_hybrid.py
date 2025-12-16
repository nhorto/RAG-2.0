#!/usr/bin/env python3
"""Migrate existing collection to hybrid (dense + sparse) embeddings."""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.qdrant_client import get_qdrant_manager
from src.ingestion.embedding_generator import HybridEmbeddingGenerator


def migrate_collection(
    old_collection: str = "consulting_transcripts",
    new_collection: str = "consulting_transcripts_hybrid",
    batch_size: int = 100,
):
    """Migrate existing dense-only collection to hybrid.

    Args:
        old_collection: Existing collection name
        new_collection: New collection name with hybrid vectors
        batch_size: Batch size for processing
    """
    print(f"Starting migration: {old_collection} → {new_collection}")

    # Initialize managers
    old_qdrant = get_qdrant_manager()
    old_qdrant.collection_name = old_collection

    new_qdrant = get_qdrant_manager()
    new_qdrant.collection_name = new_collection

    # Create new collection with sparse support
    print(f"\n1. Creating new collection '{new_collection}'...")
    success = new_qdrant.create_collection(
        collection_name=new_collection,
        enable_sparse=True
    )

    if not success:
        print("Failed to create new collection. It may already exist.")
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() == "yes":
            new_qdrant.delete_collection(new_collection)
            new_qdrant.create_collection(new_collection, enable_sparse=True)
        else:
            return

    # Get all points from old collection
    print(f"\n2. Retrieving points from '{old_collection}'...")
    all_points = []
    offset = None

    while True:
        points, offset = old_qdrant.scroll_points(
            limit=batch_size,
            offset=offset,
            with_vectors=False  # Don't need old vectors
        )

        if not points:
            break

        all_points.extend(points)
        print(f"   Retrieved {len(all_points)} points...")

        if offset is None:
            break

    print(f"\n   Total points to migrate: {len(all_points)}")

    # Generate new hybrid embeddings
    print(f"\n3. Generating hybrid embeddings...")
    embedder = HybridEmbeddingGenerator()

    texts = [point["payload"]["text"] for point in all_points]
    hybrid_embeddings = embedder.generate_embeddings(texts, show_progress=True)

    # Prepare chunks for upsert
    chunks = []
    for point in all_points:
        chunk = {
            "chunk_id": point["id"],
            **point["payload"]
        }
        chunks.append(chunk)

    # Upload to new collection
    print(f"\n4. Uploading to new collection...")
    new_qdrant.upsert_hybrid(
        chunks=chunks,
        hybrid_embeddings=hybrid_embeddings,
        batch_size=batch_size
    )

    # Verify counts
    old_info = old_qdrant.get_collection_info()
    new_info = new_qdrant.get_collection_info()

    old_count = old_info.get("points_count", 0) if old_info.get("exists") else 0
    new_count = new_info.get("points_count", 0) if new_info.get("exists") else 0

    print(f"\n5. Migration complete!")
    print(f"   Old collection: {old_count} points")
    print(f"   New collection: {new_count} points")

    if old_count == new_count:
        print("   ✅ Counts match - migration successful!")
        print(f"\nNext steps:")
        print(f"1. Test queries on '{new_collection}'")
        print(f"2. Update config to use '{new_collection}' as default")
        print(f"3. Optionally delete '{old_collection}' after verification")
    else:
        print(f"   ⚠️  Warning: Counts don't match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate collection to hybrid embeddings"
    )
    parser.add_argument(
        "--old-collection",
        default="consulting_transcripts",
        help="Existing collection name"
    )
    parser.add_argument(
        "--new-collection",
        default="consulting_transcripts_hybrid",
        help="New collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing"
    )

    args = parser.parse_args()

    migrate_collection(
        old_collection=args.old_collection,
        new_collection=args.new_collection,
        batch_size=args.batch_size
    )
