# 03_generate_star_dataset.py

import re
import json
import datasets
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_final_answer(model_output):
    """Extracts the last numerical value from a string."""
    numbers = re.findall(r'[\d,]+\.?\d*', model_output)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def main():
    """
    Generates the STaR dataset by iterating through the GSM8k train set,
    generating rationales, and using rationalization for incorrect answers.
    """
    print("Loading the original pre-trained model for STaR data generation...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Original model loaded successfully.")

    gsm8k_train_dataset = datasets.load_dataset("openai/gsm8k", 'main')['train']
    star_dataset = []
    output_file_path = "./star_dataset.jsonl"

    for example in tqdm(gsm8k_train_dataset, desc="Generating STaR Dataset"):
        question = example['question']
        ground_truth_rationale = example['answer']
        ground_truth_answer = extract_final_answer(ground_truth_rationale)

        # 1. First Attempt: Zero-Shot CoT
        zeroshot_prompt = f"{question}\n\nLet's think step by step."
        messages = [{"role": "user", "content": zeroshot_prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        model_answer = extract_final_answer(generated_text)
        
        is_correct = False
        if model_answer and ground_truth_answer:
            try:
                if float(model_answer) == float(ground_truth_answer):
                    is_correct = True
            except ValueError:
                pass 

        if is_correct:
            final_rationale = generated_text
        else:
            # 2. If incorrect, use Rationalization
            rationalization_prompt = f"Question: {question}\nHere is the correct answer: {ground_truth_answer}\n\nPlease provide a step-by-step explanation of how to arrive at this answer."
            messages = [{"role": "user", "content": rationalization_prompt}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
            final_rationale = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        star_dataset.append({"question": question, "answer": final_rationale})

    with open(output_file_path, 'w') as f:
        for entry in star_dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\n--- STaR Dataset Generation Complete ---")
    print(f"Dataset saved to: {output_file_path}")
    print(f"Total examples generated: {len(star_dataset)}")

if __name__ == "__main__":
    main()
