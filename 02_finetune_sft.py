import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from utils import extract_final_answer, ConcatDataset, train, evaluation
from dataclasses import dataclass

@dataclass
class SFT_Config:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_name: str = "openai/gsm8k"
    output_dir: str = "./models/sft_model"
    num_epochs: int = 1
    batch_size_training: int = 16
    gradient_accumulation_steps: int = 1
    context_length: int = 512
    lr: float = 2e-5
    run_validation: bool = True
    val_batch_size: int = 16
    save_model: bool = True

def get_preprocessed_dataset_sft(tokenizer, split):
    dataset = datasets.load_dataset("openai/gsm8k", 'main')[split]
    def format_prompt(sample):
        prompt = f"### Instruction:\n{sample['question']}\n\n### Response:\n"
        response = f"{sample['answer']}{tokenizer.eos_token}"
        formatted_sample = tokenizer(prompt + response, max_length=512, truncation=True)
        formatted_sample["labels"] = formatted_sample["input_ids"].copy()
        return formatted_sample
    return dataset.map(format_prompt, remove_columns=list(dataset.features))

def main():
    config = SFT_Config()
    
    # --- Model and Tokenizer Loading ---
    print(f"Loading base model for SFT: {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # --- Data Preparation ---
    train_dataset = get_preprocessed_dataset_sft(tokenizer, 'train')
    val_dataset_full = get_preprocessed_dataset_sft(tokenizer, 'test')
    val_dataset = val_dataset_full.select(range(200)) # Use a subset for validation

    packed_train = ConcatDataset(train_dataset, chunk_size=config.context_length)
    packed_val = ConcatDataset(val_dataset, chunk_size=config.context_length)

    train_dataloader = torch.utils.data.DataLoader(
        packed_train, batch_size=config.batch_size_training, shuffle=True, collate_fn=default_data_collator, drop_last=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        packed_val, batch_size=config.val_batch_size, collate_fn=default_data_collator, drop_last=True
    )

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    
    print("--- Starting Supervised Fine-Tuning (SFT) ---")
    train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, config.gradient_accumulation_steps, config)
    print("--- SFT Complete ---")

    # --- Final Evaluation ---
    # (This can be a separate script, but included here for completeness)
    print("\n--- Evaluating SFT Model on Full Test Set ---")
    eval_model_path = config.output_dir
    del model # Clear memory
    tokenizer = AutoTokenizer.from_pretrained(eval_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        eval_model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # Re-use zeroshot evaluation logic but with the SFT prompt format
    # [Evaluation logic would go here, similar to 01_evaluate_zeroshot.py]

if __name__ == "__main__":
    main()
