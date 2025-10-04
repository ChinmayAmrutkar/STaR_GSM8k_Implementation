# Implementation of "STaR: Self-Taught Reasoner" on GSM8k

This repository contains a PyTorch-based implementation of the paper [**"STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning"**](https://arxiv.org/abs/2203.14465). The project's goal is to replicate the STaR methodology on the `GSM8k` dataset using the `meta-llama/Llama--3.2-3B-Instruct` model and compare its performance against standard baselines.

This project is currently **in progress**.

## The STaR Method

The Self-Taught Reasoner (STaR) is a method for improving a language model's reasoning capabilities without requiring a large, human-annotated dataset of step-by-step explanations (rationales). The core idea is to have the model "teach itself" to reason by bootstrapping its own training data.

The process follows a simple, powerful loop:

1.  **Generate:** For a given problem, the model attempts to generate a rationale and an answer.
2.  **Check:** The generated answer is compared against the known correct answer.
3.  **Refine & Augment:**
    * If the answer is **correct**, the successful rationale is kept for training.
    * If the answer is **incorrect**, the model is given a "hint" (the correct answer) and tasked with generating a new rationale that explains how to reach that answer. This is called **rationalization**.
4.  **Fine-Tune:** A new, high-quality dataset is created from the successfully generated and rationalized examples. The original model is then fine-tuned on this bootstrapped dataset to enhance its reasoning skills.

This workflow is illustrated in the diagram from the original paper:

<img width="940" height="368" alt="image" src="https://github.com/user-attachments/assets/b620a291-4896-4ba6-9b48-9a79e1ebde9f" />

*Figure 1: An overview of the STaR workflow (Zelikman et al., 2022).*

## Project Structure
```
├── STaR_Implementation_Notebook.ipynb  # The complete development notebook with outputs
├── 01_evaluate_zeroshot.py           # Script to evaluate the base model (Baseline 1)
├── 02_finetune_sft.py                # Script to fine-tune and evaluate the vanilla SFT model (Baseline 2)
├── 03_generate_star_dataset.py       # Script to generate the bootstrapped STaR dataset
├── 04_finetune_star.py               # Script to fine-tune and evaluate the final STaR model
├── utils.py                          # Helper functions for data processing and training
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## Repository Contents

This repository is organized into two main components:

* **Python Scripts (`.py` files):** These files provide a clean, modular, and reproducible pipeline for the entire project. They are designed to be run from the command line and represent the final, polished implementation.
* **Jupyter Notebook (`.ipynb` file):** The `STaR_Implementation_Notebook.ipynb` file is the complete development log. It contains all the code from the scripts but also includes the cell-by-cell execution, detailed outputs, logs, and real-time results from the experiments. This serves as a verifiable record of the work and allows for easy inspection of the model's performance at each stage.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChinmayAmrutkar/STaR_GSM8k_Implementation.git
    cd STaR_GSM8k_Implementation
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hugging Face Authentication:**
    You will need a Hugging Face token with access to Llama 3.2 models. Log in via the terminal:
    ```bash
    huggingface-cli login
    ```

## Hardware and Configuration

This project was developed and run on a single **NVIDIA A100 GPU**. The training parameters within the scripts (e.g., `batch_size`, `context_length`) have been optimized for this hardware to ensure efficient performance. If you are using a different GPU, you may need to adjust these parameters to manage memory usage.

## How to Run

Execute the scripts in numerical order to run the full pipeline.

1.  **Run Baseline 1 (Zero-Shot CoT Evaluation):**
    ```bash
    python 01_evaluate_zeroshot.py
    ```

2.  **Run Baseline 2 (Vanilla SFT):**
    ```bash
    python 02_finetune_sft.py
    ```

3.  **Generate the STaR Dataset:**
    *This is a long-running process.*
    ```bash
    python 03_generate_star_dataset.py
    ```

4.  **Fine-Tune and Evaluate the STaR Model:**
    ```bash
    python 04_finetune_star.py
    ```

## Current Results (In Progress)

The following table shows the results achieved so far. The final STaR model evaluation is pending the completion of the dataset generation and fine-tuning steps.

| Method                    | Accuracy on GSM8k Test Set | Status         |
| :------------------------ | :------------------------- | :------------- |
| Baseline 1: Zero-Shot CoT | 63.99%                     | ✅ **Completed** |
| Baseline 2: Vanilla SFT   | 63.84%                     | ✅ **Completed** |
| **STaR Method** | TBD                        | 🔄 **In Progress** |

## Citation

This work is an implementation of the model and methodology described in the following paper:

```bibtex
@misc{zelikman2022star,
      title={STaR: Self-Taught Reasoner Bootstrapping Reasoning with Reasoning},
      author={Eric Zelikman and Yuhuai Wu and Jesse Mu and Noah D. Goodman},
      year={2022},
      eprint={2203.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
