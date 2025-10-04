# evaluate_model.py

import re
import datasets
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def extract_final_answer(model_output):
    """Extracts the last numerical value from a string."""
    numbers = re.findall(r'[\d,]+\.?\d*', model_output)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the GSM8k test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    args = parser.parse_args()

    print(f"--- Evaluating model from path: {args.model_path} ---")

    print("Loading the fine-tuned model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("Model loaded successfully.")

    gsm8k_test_dataset = datasets.load_dataset("openai/gsm8k", 'main')['test']
    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(gsm8k_test_dataset, desc=f"Evaluating {args.model_path}"):
        question = example['question']
        ground_truth_answer = extract_final_answer(example['answer'])
        
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        model_answer = extract_final_answer(response_text)

        if model_answer and ground_truth_answer:
            try:
                if float(model_answer) == float(ground_truth_answer):
                    correct_predictions += 1
            except ValueError:
                pass
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n--- Evaluation Complete for {args.model_path} ---")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
