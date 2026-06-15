import os

# Data and Core Setup
DATA_DIR: str = "./data"
CHECKPOINT_FILE: str = "./chunks_checkpoint.json"

# Pinecone Config
INDEX_NAME: str = "pharma-trial-navigator"
PINECONE_METRIC: str = "cosine"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"

# Embedding Details
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIM: int = 3072
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 200

# Agent LLM Model
AGENT_MODEL: str = "gemini-2.5-flash"

# Helper to verify key presence
def verify_environment():
    """Ensure all required API keys are available in the environment."""
    required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
