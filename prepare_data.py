import os
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

print("--- Data preprocessing script started ---")


# 1. Define functions (same as your original)
def find_files(dirs):
    files = []
    for dir_name in dirs:
        base_path = os.path.join("/iridisfs/scratch/zh1c23/Data/data/pt/", dir_name)
        if os.path.isdir(base_path):
            for dirpath, _, filenames in os.walk(base_path):
                for filename in filenames:
                    if filename.endswith(".parquet"):
                        files.append(os.path.join(dirpath, filename))
    return files


def preprocess_dataset(examples):
    eos_token = "<|im_end|>"
    text_examples = [str(text) + eos_token for text in examples["text"]]
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = 1024
    total_length = (total_length // block_size) * block_size
    result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)] for k, t in
              concatenated_examples.items()}
    return result


# 2. Define path and tokenizer
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
output_save_path = "processed_dataset_cache"

# 3. Check if already processed
if os.path.exists(output_save_path):
    print(f"Found processed dataset at '{output_save_path}', no reprocessing needed.")
else:
    print("Processed dataset not found, starting preprocessing now...")
    directories = [
        "accommodation_catering_hotel", "artificial_intelligence_machine_learning",
        "computer_communication", "computer_programming_code", "film_entertainment",
        "literature_emotion", "news_media", "tourism_geography",
        "current_affairs_government_administration", "mathematics_statistics",
    ]
    data_files = find_files(directories)
    if not data_files:
        raise FileNotFoundError("Error: No .parquet data files found under /iridisfs/scratch/zh1c23/Data/data/pt/.")

    # 4. Load and process data in a safe environment
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"])
    print("Starting map operation, this may take a long time...")
    processed_dataset = dataset.map(
        preprocess_dataset, batched=True, batch_size=5000,
        remove_columns=dataset.column_names, num_proc=16
    )

    # 5. Save the processed results to disk
    print(f"Processing complete, saving results to '{output_save_path}'...")
    processed_dataset.save_to_disk(output_save_path)

print("--- Data preprocessing script finished ---")