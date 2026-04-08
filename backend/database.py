import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import io
import os

# Production-grade embedding model
# Runs on CPU/GPU and caches in RAM for offline speed
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu" 
)

# Persistent storage: Saves to the 'chroma_db' folder
_client = chromadb.PersistentClient(path="./chroma_db")

def index_text_snippet(text: str, source: str = "manual_chat"):
    """Saves a single string to ChromaDB for long-term memory."""
    try:
        collection = _client.get_or_create_collection(
            name="user_data_vault", 
            embedding_function=_ef
        )
        doc_id = f"note_{os.urandom(4).hex()}"
        collection.add(
            documents=[text],
            metadatas=[{"source": source}],
            ids=[doc_id]
        )
        return True
    except Exception:
        return False

def index_any_csv(file_content: bytes, filename: str):
    """Production-ready indexing for structured data."""
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
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notnull(val)])
            documents.append(row_str)
            metadatas.append({"source": filename, "row_index": i})
            ids.append(f"{filename}_{i}_{os.urandom(4).hex()}") 

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return {"status": "success", "rows_indexed": len(documents)}
    except Exception as e:
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}

def search_data_vault(query: str, n_results: int = 10):
    """Retrieves relevant matches from memory for LLM context."""
    try:
        collection = _client.get_or_create_collection(name="user_data_vault", embedding_function=_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant technical data found in memory."
            
        return "\n".join(results['documents'][0])
    except Exception:
        return "Memory vault is currently initializing."
