# 04_finetune_star.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from dataclasses import dataclass
from datasets import load_dataset

# Import helper functions from utils.py
from utils import get_preprocessed_dataset_star, ConcatDataset, train

@dataclass
class STaR_Config:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_file: str = "./star_dataset.jsonl" 
    num_epochs: int = 1
    batch_size_training: int = 16
    gradient_accumulation_steps: int = 1
    context_length: int = 512
    num_workers_dataloader: int = 1
    lr: float = 2e-5
    batching_strategy: str = "packing"
    run_validation: bool = True
    val_batch_size: int = 16
    output_dir: str = "./star_model" 
    save_model: bool = True
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    weight_decay: float = 0.1
    gamma: float = 0.85
    seed: int = 42

def main():
    train_config_star = STaR_Config()

    print("Reloading the original base model for STaR fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained(train_config_star.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        train_config_star.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Original model reloaded.")

    full_star_dataset = get_preprocessed_dataset_star(tokenizer, train_config_star.dataset_file)
    train_val_split = full_star_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset_star = train_val_split['train']
    val_dataset_star = train_val_split['test']

    packed_train_dataset_star = ConcatDataset(train_dataset_star, chunk_size=train_config_star.context_length)
    packed_val_dataset_star = ConcatDataset(val_dataset_star, chunk_size=train_config_star.context_length)

    train_dataloader_star = torch.utils.data.DataLoader(
        packed_train_dataset_star,
        batch_size=train_config_star.batch_size_training,
        num_workers=train_config_star.num_workers_dataloader,
        shuffle=True,
        collate_fn=default_data_collator,
        drop_last=True
    )
    eval_dataloader_star = torch.utils.data.DataLoader(
        packed_val_dataset_star,
        batch_size=train_config_star.val_batch_size,
        num_workers=train_config_star.num_workers_dataloader,
        shuffle=False,
        collate_fn=default_data_collator,
        drop_last=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=train_config_star.lr, weight_decay=train_config_star.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config_star.gamma)

    print("--- Starting STaR Fine-Tuning ---")
    results_star = train(
        model,
        train_dataloader_star,
        eval_dataloader_star,
        tokenizer,
        optimizer,
        scheduler,
        train_config_star.gradient_accumulation_steps,
        train_config_star,
    )
    print("--- STaR SFT Complete ---")
    print(results_star)

if __name__ == "__main__":
    main()
