import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_path = "/home/hanzhuolin/PT_LORA1_Merge/"
adapter_path = "/home/hanzhuolin/PT_LO_LOSmall/checkpoint-500/"
merged_model_path = "/home/hanzhuolin/PT_LORA1sm2_Merge/"

print(f"Creating output directory: {merged_model_path}")
os.makedirs(merged_model_path, exist_ok=True)

print(f"Loading base model and tokenizer from '{base_model_path}'...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

print(f"Loading LoRA adapter from '{adapter_path}'...")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter weights into the base model...")
merged_model = peft_model.merge_and_unload()
print("Weight merge complete.")

print(f"Saving the merged model to '{merged_model_path}'...")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("="*50)
print("Success! The merged model has been saved.")
print(f"The new model is located at: {merged_model_path}")
print("="*50)