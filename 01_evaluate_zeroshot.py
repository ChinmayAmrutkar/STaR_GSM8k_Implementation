import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import extract_final_answer

def main():
    """
    Evaluates the base Llama 3.2 model on the GSM8k test set using Zero-Shot CoT.
    """
    # --- Model and Tokenizer Loading ---
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # Use compatible attention
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.")

    # --- Dataset Loading ---
    print("Loading GSM8k test dataset...")
    gsm8k_test_dataset = datasets.load_dataset("openai/gsm8k", 'main')['test']

    # --- Evaluation Loop ---
    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(gsm8k_test_dataset, desc="Evaluating Zero-Shot CoT"):
        question = example['question']
        ground_truth_answer = extract_final_answer(example['answer'])
        
        prompt = f"{question}\n\nLet's think step by step."
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        model_answer = extract_final_answer(response_text)

        if model_answer and ground_truth_answer:
            try:
                if float(model_answer) == float(ground_truth_answer):
                    correct_predictions += 1
            except ValueError:
                pass
        total_predictions += 1

    # --- Print Results ---
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n--- Zero-Shot CoT Evaluation Complete ---")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
