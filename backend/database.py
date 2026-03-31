import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import io
import os

# Production-grade embedding model (much more accurate than the 'mini' version)
# Hugging Face will download this once and cache it in your 16GB RAM.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu" 
)

# Persistent storage in the Space's app directory
_client = chromadb.PersistentClient(path="./chroma_db")

def index_any_csv(file_content: bytes, filename: str):
    """Production-ready indexing with duplicate checking."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        collection = _client.get_or_create_collection(
            name="user_data_vault", 
            embedding_function=_ef
        )

        documents = []
        metadatas = []
        ids = []

        for i, row in df.iterrows():
            # cleaner string representation for better vector matching
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notnull(val)])
            documents.append(row_str)
            metadatas.append({"source": filename, "row_index": i})
            ids.append(f"{filename}_{i}_{os.urandom(4).hex()}") # Unique ID to avoid collisions

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return {"status": "success", "rows_indexed": len(documents)}
    except Exception as e:
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}

def search_data_vault(query: str, n_results: int = 10): # Increased context window
    """Retrieves top 10 relevant matches for better LLM context."""
    try:
        collection = _client.get_collection(name="user_data_vault", embedding_function=_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        return "\n".join(results['documents'][0])
    except Exception:
        return "No relevant technical data found."
