import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHECKPOINT_FILE

def load_and_chunk_documents():
    """
    Loads all PDFs from the defined DATA_DIR and splits them into semantic chunks.
    """
    print(f"Loading PDFs from '{DATA_DIR}'...")
    loader = DirectoryLoader(DATA_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} pages across all PDF files.")

    print(f"Splitting pages into semantic chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    docs = splitter.split_documents(raw_docs)
    print(f"Result: {len(docs)} chunks.")

    return docs

def save_checkpoint(docs) -> None:
    """
    Serialize chunks to JSON so you can inspect or re-upload them manually
    if the Pinecone upsert fails. Each record contains the page content
    and all metadata returned by the loader/splitter.
    """
    records = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved -> {CHECKPOINT_FILE} ({len(records)} chunks)")
