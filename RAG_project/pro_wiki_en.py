import json
import time
import os
import torch
import gc
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. Configuration Area ---
# Input file path for the chunked data (ensure this path is correct)
CHUNKED_FILE_PATH = "/iridisfs/scratch/zh1c23/Qwen/Rag/pro_wiki/enwiki_cleaned.jsonl"
# Embedding model selection - use local path
EMBEDDING_MODEL_NAME = "/iridisfs/scratch/zh1c23/Qwen/Rag/models/models--BAAI--bge-base-en-v1.5/"
# Folder name to save the final vector database
VECTORSTORE_OUTPUT_PATH = "/iridisfs/scratch/zh1c23/Qwen/Rag/faiss_index_wiki_new/"
# Batch processing settings for CPU optimization
DOCUMENT_BATCH_SIZE = 500  # Process documents in smaller batches
EMBEDDING_BATCH_SIZE = 16  # Smaller embedding batch size for CPU

# --- 2. Data Loading Function (for .jsonl files) ---
def load_jsonl_chunks(file_path, max_docs=None):
    """Loads chunked data from a JSON Lines file and converts it to Document objects."""
    documents = []
    print(f"INFO: Starting to load data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                print(f"INFO: Reached maximum document limit of {max_docs}")
                break
                
            try:
                data = json.loads(line)
                # Assuming each line in your .jsonl file has 'text' and 'meta' keys
                # If the key names are different, please modify them here
                doc = Document(page_content=data.get('text', ''), metadata=data.get('meta', {}))
                documents.append(doc)
                
                # Progress indicator for large files
                if (i + 1) % 10000 == 0:
                    print(f"INFO: Loaded {i + 1} documents...")
                    
            except json.JSONDecodeError:
                print(f"WARNING: Line {i+1} is not valid JSON, skipping.")
    
    print(f"INFO: Successfully loaded {len(documents)} text chunks.")
    return documents

def process_documents_in_batches(documents, embedding_model, batch_size=DOCUMENT_BATCH_SIZE):
    """Process documents in batches to manage memory usage on CPU."""
    print(f"INFO: Processing {len(documents)} documents in batches of {batch_size}...")
    
    vectorstore = None
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch_num = (i // batch_size) + 1
        batch_docs = documents[i:i + batch_size]
        
        print(f"INFO: Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")
        
        if vectorstore is None:
            # Create initial vectorstore with first batch
            vectorstore = FAISS.from_documents(
                documents=batch_docs,
                embedding=embedding_model
            )
        else:
            # Add subsequent batches to existing vectorstore
            batch_vectorstore = FAISS.from_documents(
                documents=batch_docs,
                embedding=embedding_model
            )
            vectorstore.merge_from(batch_vectorstore)
            
            # Clean up memory
            del batch_vectorstore
            gc.collect()
        
        print(f"INFO: Completed batch {batch_num}/{total_batches}")
    
    return vectorstore

# --- 3. Main Program ---
if __name__ == '__main__':
    start_time = time.time()
    
    # Check if output directory already exists
    if os.path.exists(VECTORSTORE_OUTPUT_PATH):
        import shutil
        backup_path = VECTORSTORE_OUTPUT_PATH + "_backup_" + str(int(time.time()))
        print(f"WARNING: Output directory '{VECTORSTORE_OUTPUT_PATH}' already exists.")
        print(f"INFO: Moving existing directory to {backup_path}")
        shutil.move(VECTORSTORE_OUTPUT_PATH, backup_path)
    
    # Load documents
    # For testing on CPU, you might want to limit the number of documents
    # Remove max_docs parameter or set to None for full dataset
    all_chunks = load_jsonl_chunks(CHUNKED_FILE_PATH, max_docs=None)
    
    if not all_chunks:
        print("ERROR: Failed to load any text chunks, terminating program.")
        exit()
    
    # Device selection with CPU optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Using CPU - this will be significantly slower than GPU!")
        print("INFO: Applied CPU optimizations (smaller batch sizes, memory management)")
    
    # Model configuration optimized for CPU
    model_kwargs = {
        "device": device,
        "trust_remote_code": True
    }
    
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": EMBEDDING_BATCH_SIZE,  # Smaller batch size for CPU
        "show_progress_bar": True
    }
    
    # Load embedding model
    print(f"INFO: Loading embedding model from: {EMBEDDING_MODEL_NAME}")
    
    # Load the pre-downloaded model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"INFO: Successfully loaded embedding model")
    
    # Create vector database
    print("\nINFO: Starting to create and index the vector database...")
    print("INFO: This process may take a considerable amount of time on CPU...")
    
    if device == "cpu" and len(all_chunks) > DOCUMENT_BATCH_SIZE:
        # Use batch processing for CPU to manage memory
        vectorstore = process_documents_in_batches(all_chunks, embedding_model)
    else:
        # Process all at once (for GPU or small datasets)
        vectorstore = FAISS.from_documents(
            documents=all_chunks,
            embedding=embedding_model
        )
    
    print("INFO: Vector database created successfully!")
    
    # Save the vector database
    print(f"INFO: Saving the database to '{VECTORSTORE_OUTPUT_PATH}'...")
    os.makedirs(VECTORSTORE_OUTPUT_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_OUTPUT_PATH)
    print("INFO: Database saved successfully!")
    
    # Cleanup
    del vectorstore
    del embedding_model
    gc.collect()
    
    # Report completion
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"\nSUCCESS: All done! Total time taken: {total_minutes:.2f} minutes.")
    print(f"INFO: Vector database saved to: {VECTORSTORE_OUTPUT_PATH}")
    print(f"INFO: Total documents processed: {len(all_chunks)}")
    
    if device == "cpu":
        print("NOTE: Processing completed on CPU. Consider using GPU for faster processing in the future.")