
# Adaptive-Language-Modeling-for-Dynamic-Few-Shot-Tasks


This project implements a dynamic few-shot learning approach using PyTorch and Transformers, focusing on solving mathematical problems such as addition, subtraction, multiplication, and division. The model leverages GPT-2 for language modeling and prompts tailored to specific tasks, allowing adaptation to new mathematical operations with minimal training data.

## Features

- **Dynamic Memory:** Utilizes a memory module to store embeddings of encountered tasks and adapt to new tasks dynamically.
- **Prompt Tuning:** Implements prompt tuning techniques to enhance model performance on mathematical problem-solving tasks.
- **Dataset Generation:** Generates synthetic mathematical tasks (addition, subtraction, multiplication, division) with varying operands and solutions.
- **Integration with Hugging Face Hub:** Supports model checkpointing and sharing via the Hugging Face Hub for easy replication and deployment.

## Setup

### Prerequisites

- Python 3.6+
- PyTorch
- Transformers
- PyTorch Lightning
- Hugging Face Hub
- dotenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sanowl/Adaptive-Language-Modeling-for-Dynamic-Few-Shot-Tasks.git
   cd dynamic_few_shot_learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following content:
   ```
   HF_REPO_ID=your_huggingface_username
   HF_TOKEN=your_huggingface_token
   ```

### Usage

To train the model, run:
```bash
python main.py
```

## Configuration

Adjust the configuration parameters in `main.py` to customize the model behavior:
- `model_name`: Pretrained model name from Hugging Face model hub.
- `num_tokens`: Number of tokens in the prompt for task-specific tuning.
- `num_tasks`: Number of different mathematical tasks (e.g., addition, subtraction).
- `learning_rate`: Learning rate for optimizer.
- `batch_size`: Batch size for training.
- `num_epochs`: Number of training epochs.
- `samples_per_task`: Number of samples per mathematical task.
- `max_seq_length`: Maximum sequence length for tokenization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
