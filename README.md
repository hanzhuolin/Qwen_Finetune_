# Dataset Download Links

## Pre-training Dataset
- [IndustryCorpus2 (BAAI)](https://www.modelscope.cn/datasets/BAAI/IndustryCorpus2)

## Fine-tuning Datasets
- [Infinity-Instruct (BAAI)](https://www.modelscope.cn/datasets/BAAI/Infinity-Instruct)
- [blossom-orca-v3 (Hugging Face)](https://huggingface.co/datasets/Azure99/blossom-orca-v3/blob/main/README.md)

## RAG Dataset
- [Wikipedia Dumps (English)](https://dumps.wikimedia.org/enwiki/)

---

# Runtime Requirements

##Please install the following Python packages before running Pre-train and LoRA:

```bash
pip install flash-attn
pip install trl==0.11.4
pip install transformers==4.45.0

##RAG application requriment is below
torch
transformers
accelerate
bitsandbytes
safetensors
pandas
scikit-learn
datasets
faiss-gpu
sentence-transformers
chromadb
langchain
langchain-community
streamlit
beautifulsoup4

# trained model and FAISS index donload address
https://drive.google.com/drive/folders/11Lv_0flwVdm-n-InRf6KSnaeLlhflIrI?usp=drive_link
