import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import io

# Initialize Persistent Client
client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()

def index_any_csv(file_content: bytes, filename: str):
    """
    Dynamically indexes any CSV by converting rows into text descriptions.
    """
    # Load the data
    df = pd.read_csv(io.BytesIO(file_content))
    
    # Create or get a collection for this specific user/session
    collection = client.get_or_create_collection(
        name="user_data_vault", 
        embedding_function=default_ef
    )

    documents = []
    metadatas = []
    ids = []

    # Iterate through rows and build a generic description
    for i, row in df.iterrows():
        # Optimization: Create a string of 'Column: Value' pairs
        row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
        
        documents.append(row_str)
        metadatas.append({"source": filename, "row_index": i})
        ids.append(f"{filename}_{i}")

    # Index into ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return {"status": "success", "rows_indexed": len(documents), "columns": list(df.columns)}

def search_data_vault(query: str, n_results: int = 5):
    """
    Retrieves the most relevant rows from the database based on the user's question.
    """
    try:
        collection = client.get_collection(name="user_data_vault", embedding_function=default_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        # Flatten the results for the LLM context
        return "\n".join(results['documents'][0])
    except:
        return "No relevant data found in the vault."
