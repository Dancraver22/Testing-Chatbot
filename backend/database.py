import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import io

# 1. FIX: Use a specific, lightweight embedding function
# This model is only ~80MB, perfect for Render's 512MB limit.
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"  # Force CPU to avoid Torch/CUDA memory errors
)

# Initialize Persistent Client
client = chromadb.PersistentClient(path="./chroma_db")

def index_any_csv(file_content: bytes, filename: str):
    """Dynamically indexes CSV rows into ChromaDB."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Use the explicit embedding function here
        collection = client.get_or_create_collection(
            name="user_data_vault", 
            embedding_function=ef
        )

        documents = []
        metadatas = []
        ids = []

        for i, row in df.iterrows():
            # Create a searchable string: "Column: Value | Column: Value"
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(row_str)
            metadatas.append({"source": filename, "row_index": i})
            ids.append(f"{filename}_{i}")

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return {"status": "success", "rows_indexed": len(documents), "columns": list(df.columns)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_data_vault(query: str, n_results: int = 5):
    """Retrieves relevant rows for the LLM context."""
    try:
        collection = client.get_collection(name="user_data_vault", embedding_function=ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        return "\n".join(results['documents'][0])
    except:
        return "No relevant data found in the vault."
