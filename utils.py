# utils.py

import re
import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from contextlib import nullcontext

def extract_final_answer(model_output):
    """Extracts the last numerical value from a string."""
    numbers = re.findall(r'[\d,]+\.?\d*', model_output)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def get_preprocessed_dataset_sft(tokenizer, split):
    """Loads and preprocesses the GSM8k dataset for SFT."""
    dataset = datasets.load_dataset("openai/gsm8k", 'main')[split]
    def format_prompt(sample):
        prompt = f"### Instruction:\n{sample['question']}\n\n### Response:\n"
        response = f"{sample['answer']}{tokenizer.eos_token}"
        formatted_sample = tokenizer(prompt + response)
        formatted_sample["labels"] = formatted_sample["input_ids"].copy()
        return formatted_sample
    processed_dataset = dataset.map(format_prompt, remove_columns=list(dataset.features))
    return processed_dataset

def get_preprocessed_dataset_star(tokenizer, file_path):
    """Loads and preprocesses the STaR dataset from a JSONL file."""
    dataset = datasets.load_dataset('json', data_files=file_path, split='train')
    def format_prompt_star(sample):
        prompt = f"### Instruction:\n{sample['question']}\n\n### Response:\n"
        response = f"{sample['answer']}{tokenizer.eos_token}"
        formatted_sample = tokenizer(prompt + response)
        formatted_sample["labels"] = formatted_sample["input_ids"].copy()
        return formatted_sample
    processed_dataset = dataset.map(format_prompt_star, remove_columns=list(dataset.features))
    return processed_dataset

class ConcatDataset(Dataset):
    """Dataset for packing examples into fixed-length chunks."""
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        for sample in tqdm(self.dataset, desc="Packing dataset"):
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k, v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config):
    """Main training loop."""
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(colour="blue", desc=f"Fine-Tuning Epoch: {epoch+1}", total=len(train_dataloader), dynamic_ncols=True)
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            with nullcontext():
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss.detach().float()
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if train_config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.detach().float():.4f}")

        pbar.close()
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Train Loss: {train_epoch_loss:.4f}")

        if train_config.run_validation:
            eval_epoch_loss = evaluation(model, eval_dataloader)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.save_model:
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    model.save_pretrained(train_config.output_dir)
                    tokenizer.save_pretrained(train_config.output_dir)
    results['avg_train_loss'] = train_epoch_loss
    if train_config.run_validation:
        results['avg_eval_loss'] = best_val_loss
    return results

def evaluation(model, eval_dataloader):
    """Main evaluation loop for validation."""
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, colour="green", desc="Evaluating"):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    print(f"Validation Loss: {eval_epoch_loss:.4f}")
    return eval_epoch_loss

