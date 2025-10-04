import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import default_data_collator
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from contextlib import nullcontext
import time
import os

# --- Helper function for answer extraction ---
def extract_final_answer(model_output):
    """
    Extracts the last numerical value from the model's output.
    """
    # Use regex to find all numbers (including decimals and commas) in the output
    numbers = re.findall(r'[\d,]+\.?\d*', model_output)
    if numbers:
        # Return the very last number found, removing any commas
        return numbers[-1].replace(',', '')
    return None

# --- Dataset Packing Class ---
class ConcatDataset(Dataset):
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

# --- Training and Evaluation Functions ---
def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config):
    """
    Main training loop.
    """
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        model.train()
        total_loss = 0.0
        # Correctly calculate total steps for tqdm
        total_steps = len(train_dataloader) // gradient_accumulation_steps
        pbar = tqdm(colour="blue", desc=f"Fine-Tuning Epoch: {epoch+1}", total=total_steps, dynamic_ncols=True)

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
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.detach().float() * gradient_accumulation_steps:.4f}")

        pbar.close()
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Train Loss: {train_epoch_loss:.4f}")

        if train_config.run_validation:
            eval_epoch_loss = evaluation(model, eval_dataloader)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.save_model:
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {train_config.output_dir}")
                    model.save_pretrained(train_config.output_dir)
                    tokenizer.save_pretrained(train_config.output_dir)
    return {"final_train_loss": train_epoch_loss, "best_val_loss": best_val_loss}

def evaluation(model, eval_dataloader):
    """
    Main evaluation loop.
    """
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
