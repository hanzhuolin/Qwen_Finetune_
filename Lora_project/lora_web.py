import os
import json
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODIFIED: Separated model output path and loss output path ---
model_output_path = "/iridisfs/scratch/zh1c23/Qwen/Lora/LOSS/PT_lora2_loss/model/"
loss_output_path = "/iridisfs/scratch/zh1c23/Qwen/Lora/LOSS/PT_lora2_loss/loss/"  # <--- NEW: Dedicated save path for loss files
model_path ="/iridisfs/scratch/zh1c23/Qwen/results/pt_final_model/"

# --- MODIFIED: Ensure both output directories exist ---
os.makedirs(model_output_path, exist_ok=True)
os.makedirs(loss_output_path, exist_ok=True)  # <--- NEW: Create loss output directory

class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.save_directory = os.path.dirname(log_file_path)
        # For saving loss at each step
        self.step_log_file_path = log_file_path
        self.step_loss_history = []
        
        # For saving average loss at the end of each epoch
        self.epoch_log_file_path = os.path.join(self.save_directory, "epoch_loss_history.json")
        self.epoch_loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called at each logging step to save loss."""
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            learning_rate = logs.get("learning_rate", 0)
            
            # --- CORE: Print loss to console in real-time ---
            print(f"Step {step:6d} | Loss: {loss:.6f} | LR: {learning_rate:.2e}")
            
            self.step_loss_history.append({
                "step": step,
                "loss": loss,
                "learning_rate": learning_rate
            })
            
            self.save_step_loss_history()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch to calculate average loss and plot the loss curve."""
        epoch = int(state.epoch)
        print(f"Epoch {epoch} finished. Calculating average loss for this epoch...")

        # NOTE: This log_history is built into the Trainer and may differ slightly from our manually recorded step_loss_history
        epoch_logs = [
            log for log in state.log_history 
            if log.get("epoch") is not None and int(log["epoch"]) == epoch and "loss" in log
        ]
        
        if epoch_logs:
            avg_loss = sum(log["loss"] for log in epoch_logs) / len(epoch_logs)
            
            self.epoch_loss_history.append({
                "epoch": epoch,
                "average_loss": avg_loss,
                "total_steps_in_epoch": len(epoch_logs)
            })
            
            print(f"Epoch {epoch} | Average Loss: {avg_loss:.6f}")
            self.save_epoch_loss_history()

        print(f"Plotting loss curve after Epoch {epoch}...")
        self._plot_loss_curve(current_epoch=epoch)

    def _plot_loss_curve(self, current_epoch=None):
        """
        Internal method to plot the loss curve.
        If current_epoch is provided, the filename will include the epoch number.
        """
        if not self.step_loss_history:
            print("step_loss_history is empty, cannot plot the loss curve.")
            return
            
        plt.switch_backend("agg")
        
        steps = [entry["step"] for entry in self.step_loss_history]
        losses = [entry["loss"] for entry in self.step_loss_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, color="#1f77b4", label="Training Loss", linewidth=2)
        
        title = "Training Loss Curve"
        if current_epoch:
            title = f"{title} after Epoch {current_epoch}"
        
        plt.title(title, fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = "training_loss_final.png"
        if current_epoch:
            filename = f"training_loss_epoch_{current_epoch}.png"
        
        figure_path = os.path.join(self.save_directory, filename)
        plt.savefig(figure_path, format="png", dpi=150, bbox_inches='tight')
        print(f"Training loss curve saved to: {figure_path}")
        plt.close()

    def save_step_loss_history(self):
        """Saves the step-by-step loss history."""
        try:
            with open(self.step_log_file_path, 'w') as f:
                json.dump(self.step_loss_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save step loss history: {e}")
            
    def save_epoch_loss_history(self):
        """Saves the epoch-by-epoch loss history."""
        try:
            with open(self.epoch_log_file_path, 'w') as f:
                json.dump(self.epoch_loss_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save epoch loss history: {e}")

def load_model_and_tokenizer():
    """Load model and tokenizer"""
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    
    model.config.repetition_penalty = 1.2
    model.config.no_repeat_ngram_size = 3
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def find_files(dirs):
    """Find data files"""
    files = []
    for dir in dirs:
        base_path = os.path.join("/iridisfs/scratch/zh1c23/Data/data/sft/", dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

def load_and_prepare_dataset():
    """Load and prepare dataset with train/eval split"""
    print("Loading dataset...")
    directories = ["7M", "Gen"]
    data_files = find_files(directories)
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["conversations"])
    dataset = dataset.shuffle(seed=42)

    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Train dataset loaded with {len(train_dataset)} samples")
    print(f"Eval dataset loaded with {len(eval_dataset)} samples")
    
    return train_dataset, eval_dataset

def formatting_prompts_func(example):
    """Format prompts function"""
    output_texts = []
    for i in range(len(example["conversations"])):
        human_text = ""
        gpt_text = ""
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def setup_lora_config():
    """Setup enhanced LoRA configuration"""
    return LoraConfig(
        r=128,  
        lora_alpha=256,  
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.1,  
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]  
    )

def setup_training_args():
    """Setup optimized training arguments"""
    return SFTConfig(
        output_dir=model_output_path,
        overwrite_output_dir=False,
        learning_rate=1e-5,  
        warmup_ratio=0.1,  
        lr_scheduler_type="cosine_with_restarts",  
        num_train_epochs=3,  
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=32,  
        per_device_eval_batch_size=8,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        eval_strategy="steps",  
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        logging_steps=20,
        max_seq_length=1024,
        packing=False,
        dataset_num_proc=16,
        dataset_batch_size=5000,
        resume_from_checkpoint=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  
        optim="adamw_8bit",  
        weight_decay=0.05,  
        max_grad_norm=0.5,  
        seed=42,
    )

def check_and_resume_training():
    """Check and resume training"""
    last_checkpoint = get_last_checkpoint(model_output_path)
    if last_checkpoint is not None:
        print(f"Checkpoint found, resuming training from '{last_checkpoint}'")
        return last_checkpoint
    else:
        print("No valid checkpoint found, starting a new training run")
        return None

def main():
    """Main training function"""
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)
    
    lora_config = setup_lora_config()
    training_args = setup_training_args()
    log_file_path = os.path.join(loss_output_path, "loss_history.json")
    loss_callback = LossLoggingCallback(log_file_path)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=lora_config,
        callbacks=[loss_callback]
    )
    
    last_checkpoint = check_and_resume_training()
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save the final model to the model output path
    final_model_save_path = os.path.join(model_output_path, "final_lora_adapter")
    trainer.save_model(final_model_save_path)
    print(f"Final LoRA adapter saved successfully to: {final_model_save_path}")
    
    tokenizer.save_pretrained(final_model_save_path)
    print(f"Tokenizer saved successfully to: {final_model_save_path}")
    
    print("Plotting the final loss curve for the entire training...")
    loss_callback._plot_loss_curve()
    
    final_stats = {
        "total_steps": trainer.state.global_step,
        "total_epochs": trainer.state.epoch,
        "final_loss": loss_callback.step_loss_history[-1]["loss"] if loss_callback.step_loss_history else None,
        "best_eval_loss": trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None,
        "training_completed": True
    }
    
    # --- MODIFIED: Training stats are also saved to the loss path ---
    with open(os.path.join(loss_output_path, "training_stats.json"), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print("Training completed!")
    print("\n=== Important: Inference Configuration ===")
    print("When using the trained model, always use these parameters:")
    print("- repetition_penalty: 1.2")
    print("- no_repeat_ngram_size: 3")
    print("- temperature: 0.7-0.8")
    print("- top_p: 0.9")
    print("- top_k: 50")

if __name__ == "__main__":
    main()