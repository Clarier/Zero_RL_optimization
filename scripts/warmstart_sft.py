"""
Lightweight Warm-Start SFT Script
===================================
Performs 1-2 epochs of SFT on very easy countdown problems
to bootstrap weak models (e.g., Qwen2.5-0.5B) before RL.

Key design choices:
- Uses VERY few samples (50-200) to avoid overfitting
- Only trains on easy 2-number problems with known solutions
- Runs for 1-2 epochs max
- Preserves the base model's generality

This addresses the core problem: weak models produce all-zero rewards
in pure zero-RL, so GRPO cannot learn. A minimal warm-start gives
the model just enough capability to occasionally produce non-zero
rewards, kick-starting the RL loop.

Usage:
    python warmstart_sft.py \
        --model_path Qwen/Qwen2.5-0.5B \
        --data_path ./data/warmstart.parquet \
        --output_dir ./checkpoints/warmstart-0.5b \
        --n_samples 100 \
        --epochs 2 \
        --lr 5e-6
"""

import os
import json
import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import pandas as pd

logger = logging.getLogger(__name__)


class WarmStartDataset(Dataset):
    """Simple SFT dataset from warm-start parquet file."""
    
    def __init__(self, parquet_path, tokenizer, max_length=512, n_samples=None):
        df = pd.read_parquet(parquet_path)
        
        if n_samples is not None and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=42)
        
        self.examples = []
        
        for _, row in df.iterrows():
            prompt_data = json.loads(row['prompt'])
            prompt_text = prompt_data[0]['content'] if isinstance(prompt_data, list) else prompt_data
            response_text = row.get('response', '')
            
            full_text = prompt_text + response_text
            
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            
            # Create labels: mask the prompt, only train on response
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
            )
            prompt_len = len(prompt_tokens['input_ids'])
            
            labels = tokens['input_ids'].clone().squeeze()
            labels[:prompt_len] = -100  # Mask prompt tokens
            
            self.examples.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': labels,
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def run_warmstart_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    n_samples: int = 100,
    epochs: int = 2,
    lr: float = 5e-6,
    batch_size: int = 8,
    max_length: int = 512,
    gradient_accumulation_steps: int = 4,
):
    """Run warm-start SFT."""
    
    print(f"=" * 60)
    print(f"Warm-Start SFT for Negative Sample Mining")
    print(f"=" * 60)
    print(f"Model: {model_path}")
    print(f"Data:  {data_path}")
    print(f"Samples: {n_samples}")
    print(f"Epochs: {epochs}")
    print(f"LR: {lr}")
    print(f"=" * 60)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load dataset
    dataset = WarmStartDataset(
        data_path, tokenizer, max_length=max_length, n_samples=n_samples
    )
    print(f"Loaded {len(dataset)} warm-start examples")
    
    # Training arguments — very conservative
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,
        report_to='wandb',
        run_name=f'warmstart-{os.path.basename(model_path)}-n{n_samples}',
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nWarm-start model saved to {output_dir}")
    print(f"Use this as BASE_MODEL for RL training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    args = parser.parse_args()
    run_warmstart_sft(**vars(args))


if __name__ == '__main__':
    main()
