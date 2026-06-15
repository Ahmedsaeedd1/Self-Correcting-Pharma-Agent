import os
import time
import requests
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from core.config import (
    INDEX_NAME,
    EMBEDDING_DIM,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from ingestion.embedder import GeminiEmbeddings

def _create_index_via_rest(name: str, dimension: int, metric: str) -> None:
    """
    Creates a Pinecone serverless index via the REST API directly.
    The Python SDK has a bug in some versions where the deletion_protection
    parameter fails type validation regardless of what value is passed.
    Calling the REST API directly sidesteps this entirely.
    """
    url = "https://api.pinecone.io/indexes"
    headers = {
        "Api-Key": os.environ["PINECONE_API_KEY"],
        "Content-Type": "application/json",
    }
    payload = {
        "name": name,
        "dimension": dimension,
        "metric": metric,
        "spec": {
            "serverless": {
                "cloud": PINECONE_CLOUD,
                "region": PINECONE_REGION,
            }
        },
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Index creation failed ({resp.status_code}): {resp.text}")

def ensure_index() -> Pinecone:
    """
    Connects to Pinecone. Creates the index if it does not exist,
    or recreates it if the dimension is incorrect.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    print(f"Existing indexes: {existing_indexes or '(none)'}")

    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Creating (dim={EMBEDDING_DIM})...")
        _create_index_via_rest(INDEX_NAME, EMBEDDING_DIM, PINECONE_METRIC)
        print(f"Waiting 15s for index to become ready...")
        time.sleep(15)
        print(f"Index '{INDEX_NAME}' created.")
    else:
        # Verify the existing index has the right dimension
        idx_info = pc.describe_index(INDEX_NAME)
        idx_dim = idx_info.dimension
        if idx_dim != EMBEDDING_DIM:
            print(f"[WARNING] Existing index has dim {idx_dim} but needs {EMBEDDING_DIM}.")
            print(f"Deleting and recreating index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            time.sleep(5)
            _create_index_via_rest(INDEX_NAME, EMBEDDING_DIM, PINECONE_METRIC)
            print(f"Waiting 15s for new index to become ready...")
            time.sleep(15)
            print(f"Index recreated with dimension {EMBEDDING_DIM}.")
        else:
            print(f"Index '{INDEX_NAME}' already exists with correct dimension. Skipping creation.")

    return pc

def upload_chunks(pc: Pinecone, docs, batch_size=32) -> PineconeVectorStore:
    """
    Uploads document chunks to Pinecone in batches.
    Includes rate-limit retry logic for Gemin embedding endpoint.
    """
    index = pc.Index(INDEX_NAME)
    embeddings = GeminiEmbeddings()
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    total = len(docs)
    uploaded = 0
    num_batches = (total + batch_size - 1) // batch_size

    print(f"Uploading {total} chunks to '{INDEX_NAME}' in {num_batches} batches...")

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = docs[start : start + batch_size]
        max_retries = 5

        for attempt in range(1, max_retries + 1):
            try:
                vectorstore.add_documents(batch)
                uploaded += len(batch)
                print(f"Batch {batch_num:>3}/{num_batches} ({uploaded}/{total} chunks uploaded)")
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < max_retries:
                    wait = 15 * attempt
                    import re
                    match = re.search(r"retry[^\\d]*(\d+(?:\.\d+)?)\s*s", err_str, re.IGNORECASE)
                    if match:
                        wait = int(float(match.group(1))) + 2
                    print(f"  [Rate limit] Batch {batch_num} attempt {attempt}/{max_retries} — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Batch {batch_num} failed at chunk {start}-{start+len(batch)}: {e}")
                    raise

        if batch_num < num_batches:
            time.sleep(2)

    time.sleep(3)
    stats = index.describe_index_stats()
    vector_count = stats.get("total_vector_count", "n/a")
    print(f"\nAll batches done. Vectors confirmed in index: {vector_count}")

    return vectorstore
