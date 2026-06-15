import os
import sys

# Ensure the core project is imported properly if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from core.config import verify_environment, CHECKPOINT_FILE, INDEX_NAME
from ingestion.loader import load_and_chunk_documents, save_checkpoint
from ingestion.indexer import ensure_index, upload_chunks

def run_ingestion():
    """
    Main entry point for pulling PDFs, chunking them, and uploading to Pinecone.
    """
    load_dotenv()
    verify_environment()
    
    print("-" * 50)
    print("Starting Ingestion Pipeline")
    print("-" * 50)

    # 1. Load and segment the local PDFs
    docs = load_and_chunk_documents()

    # 2. Save a checkpoint in case the remote API fails halfway
    save_checkpoint(docs)

    # 3. Ensure Pinecone Database is set up
    pc = ensure_index()

    # 4. Upload standard embeddings in batches
    upload_chunks(pc, docs)

    print("\n" + "=" * 50)
    print("  Pipeline finished successfully.")
    print(f"  Index name   : {INDEX_NAME}")
    print(f"  Chunks stored: {len(docs)}")
    print(f"  Recovery file: {CHECKPOINT_FILE}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_ingestion()
