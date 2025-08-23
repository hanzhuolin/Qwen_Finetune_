import os
import shutil
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from accelerate import Accelerator

def process_and_embed_batch(docs_batch, text_splitter, embedding_model):
    """Splits and embeds a batch of documents."""
    split_docs = text_splitter.split_documents(docs_batch)
    if not split_docs:
        return None
    db = FAISS.from_documents(split_docs, embedding_model)
    return db

def create_faiss_index_disk_based(jsonl_path, embedding_model_path, save_path, chunk_size, chunk_overlap, batch_size, segment_size):
    """
    Processes a large JSONL file by creating and saving index segments to disk to keep memory usage low,
    then merges them in a final step.
    """
    try:
        accelerator = Accelerator()
        print(f"[Process {accelerator.process_index}] Initialized. World size: {accelerator.num_processes}")

        print(f"[*] [Process {accelerator.process_index}] Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': accelerator.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"[+] [Process {accelerator.process_index}] Embedding model loaded.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        print(f"[*] [Process {accelerator.process_index}] Starting to process file in segments of {segment_size} lines...")
        
        segment_db = None
        batch_docs = []
        lines_in_segment = 0
        segment_index = 0
        total_lines_processed = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % accelerator.num_processes == accelerator.process_index:
                    try:
                        text = json.loads(line).get("text", "")
                        if text:
                            batch_docs.append(Document(page_content=text))
                        
                        if len(batch_docs) >= batch_size:
                            batch_db = process_and_embed_batch(batch_docs, text_splitter, embedding_model)
                            if batch_db:
                                if segment_db is None: segment_db = batch_db
                                else: segment_db.merge_from(batch_db)
                            lines_in_segment += len(batch_docs)
                            total_lines_processed += len(batch_docs)
                            batch_docs = []

                        if segment_db and lines_in_segment >= segment_size:
                            temp_segment_path = f"{save_path}_temp_part_{accelerator.process_index}_seg_{segment_index}"
                            print(f"    - [Process {accelerator.process_index}] Saving segment {segment_index} ({lines_in_segment} docs) to disk at '{temp_segment_path}'...")
                            segment_db.save_local(temp_segment_path)
                            segment_index += 1
                            segment_db = None
                            lines_in_segment = 0
                    except json.JSONDecodeError:
                        continue
        
        if batch_docs:
            batch_db = process_and_embed_batch(batch_docs, text_splitter, embedding_model)
            if batch_db:
                if segment_db is None: segment_db = batch_db
                else: segment_db.merge_from(batch_db)
            total_lines_processed += len(batch_docs)

        if segment_db:
            temp_segment_path = f"{save_path}_temp_part_{accelerator.process_index}_seg_{segment_index}"
            print(f"    - [Process {accelerator.process_index}] Saving final segment {segment_index} to disk...")
            segment_db.save_local(temp_segment_path)

        print(f"[+] [Process {accelerator.process_index}] Finished processing. Total documents handled: {total_lines_processed}.")
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"\n[*] [Main Process] Merging all saved segment indexes...")
            
            final_db = None
            for i in range(accelerator.num_processes):
                seg_idx = 0
                while True:
                    part_path = f"{save_path}_temp_part_{i}_seg_{seg_idx}"
                    if not os.path.exists(part_path):
                        break
                    
                    print(f"    - Merging index from: {part_path}")
                    partial_db = FAISS.load_local(part_path, embedding_model, allow_dangerous_deserialization=True)
                    if final_db is None:
                        final_db = partial_db
                    else:
                        final_db.merge_from(partial_db)
                    seg_idx += 1

            if final_db:
                os.makedirs(save_path, exist_ok=True)
                final_db.save_local(save_path)
                print(f"\n[SUCCESS] Final merged FAISS index has been saved at: {save_path}")
                print(f"    - Total vectors in final index: {final_db.index.ntotal}")
            else:
                print("[ERROR] No partial indexes were created.")

            print("\n    - Cleaning up temporary files...")
            for i in range(accelerator.num_processes):
                seg_idx = 0
                while True:
                    part_path = f"{save_path}_temp_part_{i}_seg_{seg_idx}"
                    if not os.path.exists(part_path):
                        break
                    shutil.rmtree(part_path)
                    seg_idx += 1
            print("    - Cleanup complete.")

    except Exception as e:
        print(f"\n[ERROR on Process {accelerator.process_index}] An error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    SOURCE_JSONL_PATH = "/iridisfs/scratch/zh1c23/Qwen/Rag/pro_wiki/enwiki_cleaned.jsonl"
    EMBEDDING_MODEL_PATH = "/iridisfs/scratch/zh1c23/Qwen/Rag/bge-base-en-v1.5/"
    INDEX_SAVE_PATH = "/scratch/zh1c23/Qwen/Rag/faiss_index_jsonl_chunked"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    BATCH_SIZE = 2000 # The size of small in-memory batches
    SEGMENT_SIZE = 100000 # Number of documents to process before saving to disk
    
    # --- Execution ---
    create_faiss_index_disk_based(
        jsonl_path=SOURCE_JSONL_PATH,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        save_path=INDEX_SAVE_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        batch_size=BATCH_SIZE,
        segment_size=SEGMENT_SIZE
    )
