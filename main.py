import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Dict, Optional
from dataclasses import dataclass
import random
import math
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder

# Load environment variables
load_dotenv()

# Set environment variables directly for testing purposes
os.environ['HF_REPO_ID'] = 'san122'
os.environ['HF_TOKEN'] = ''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str
    num_tokens: int
    num_tasks: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    samples_per_task: int
    max_seq_length: int
    repo_name: str

class DynamicMemory(nn.Module):
    def __init__(self, num_tasks: int, num_tokens: int, embedding_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.memory = nn.Parameter(torch.randn(num_tasks, num_tokens * embedding_dim) / math.sqrt(embedding_dim))
        self.forgetting_factor = nn.Parameter(torch.tensor(0.9))

    def update(self, task_id: int, embedding: torch.Tensor) -> None:
        with torch.no_grad():
            self.memory[task_id] = self.forgetting_factor * self.memory[task_id] + (1 - self.forgetting_factor) * embedding.view(-1)

    def get(self, task_id: int) -> torch.Tensor:
        return self.memory[task_id].view(self.num_tokens, self.embedding_dim)

class DynamicPromptTuning(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gpt = AutoModelForCausalLM.from_pretrained(config.model_name)
        hidden_size = self.gpt.config.hidden_size
        self.prompt_embedding = nn.Parameter(torch.randn(config.num_tokens, hidden_size) / math.sqrt(hidden_size))
        self.task_layer = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.task_layer.weight)
        nn.init.zeros_(self.task_layer.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, task_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.shape[0]

        # Expand prompt embeddings to match batch size
        prompt_embeddings = self.prompt_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.gpt.get_input_embeddings()(input_ids)

        # Concatenate prompt embeddings and input embeddings
        combined_embeds = torch.cat((prompt_embeddings, inputs_embeds), dim=1)

        # Adjust attention mask to match the combined embeddings
        if attention_mask is not None:
            prompt_attention = torch.ones(batch_size, self.config.num_tokens, device=attention_mask.device)
            attention_mask = torch.cat((prompt_attention, attention_mask), dim=1)

        # Adjust labels to match the combined embeddings
        if labels is not None:
            prompt_labels = torch.full((batch_size, self.config.num_tokens), -100, device=labels.device)
            labels = torch.cat((prompt_labels, labels), dim=1)

        return self.gpt(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels)

class DynamicMetaLearner(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = DynamicPromptTuning(config)
        hidden_size = self.model.gpt.config.hidden_size
        self.memory = DynamicMemory(config.num_tasks, config.num_tokens, hidden_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        task_id = batch['task_id']
        if isinstance(task_id, torch.Tensor):
            task_id = task_id.item()
        task_embedding = self.memory.get(task_id)
        outputs = self.model(input_ids=batch['input']['input_ids'], 
                             attention_mask=batch['input']['attention_mask'], 
                             labels=batch['input']['labels'], 
                             task_embedding=task_embedding)
        return outputs.loss

    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        return [optimizer], [scheduler]

    def on_train_end(self):
        self.push_to_hub()

    def push_to_hub(self):
        hf_username = os.getenv('HF_REPO_ID')
        hf_token = os.getenv('HF_TOKEN')

        if not hf_username or not hf_token:
            logger.warning("Hugging Face credentials not found in environment variables. Skipping push to hub.")
            return

        # Correctly format the repo_id
        repo_id = f"{hf_username}/{self.config.repo_name}"

        api = HfApi()
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,
            exist_ok=True
        )

        # Clone the repository if it doesn't exist locally
        if not os.path.exists("./hf_model"):
            os.makedirs("./hf_model")
        
        model_path = "./hf_model"
        
        # Save the model and other files
        self.model.gpt.save_pretrained(model_path)
        torch.save(self.model.prompt_embedding, os.path.join(model_path, "prompt_embedding.pt"))
        torch.save(self.model.task_layer.state_dict(), os.path.join(model_path, "task_layer.pt"))
        
        # Upload the files to the repository
        upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            path_in_repo=".",
            token=hf_token,
            commit_message="Update model"
        )

        logger.info(f"Model pushed to Hugging Face Hub: {repo_url}")

class MathProblemDataset(Dataset):
    def __init__(self, config: ModelConfig, tokenizer):
        self.config = config
        self.tasks = self._generate_tasks()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        task = self.tasks[idx]
        sample = random.choice(task['samples'])
        return {
            'input': self._tokenize_sample(sample),
            'task_id': task['task_id']
        }

    def _tokenize_sample(self, sample: Dict[str, str]) -> Dict[str, torch.Tensor]:
        input_text = f"Problem: {sample['input']}\nSolution:"
        input_tokens = self.tokenizer(input_text, padding='max_length', truncation=True, 
                                      max_length=self.config.max_seq_length, return_tensors='pt')
        labels = self.tokenizer(sample['output'], padding='max_length', truncation=True, 
                                max_length=self.config.max_seq_length, return_tensors='pt').input_ids
        input_tokens['labels'] = labels
        return {k: v.squeeze(0) for k, v in input_tokens.items()}

    def _generate_tasks(self) -> List[Dict[str, List[Dict[str, str]]]]:
        operations = ['+', '-', '*', '/']
        tasks = []
        for task_id in range(self.config.num_tasks):
            operation = random.choice(operations)
            samples = []
            for _ in range(self.config.samples_per_task):
                a, b = random.randint(1, 100), random.randint(1, 100)
                problem, solution = self._create_problem(operation, a, b)
                samples.append({"input": problem, "output": solution})
            tasks.append({"task_id": task_id, "samples": samples})
        return tasks

    @staticmethod
    def _create_problem(operation: str, a: int, b: int) -> tuple:
        if operation == '+':
            return f"{a} + {b}", str(a + b)
        elif operation == '-':
            return f"{a} - {b}", str(a - b)
        elif operation == '*':
            return f"{a} * {b}", str(a * b)
        else:
            return f"{a*b} / {b}", str(a)

class GPT2Wrapper(nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.logits  # Only return the logits for tracing

def train_model(config: ModelConfig) -> None:
    pl.seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = MathProblemDataset(config, tokenizer)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = DynamicMetaLearner(config)
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='cpu',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(monitor='val_loss')
        ],
        logger=TensorBoardLogger("logs", name="dynamic_few_shot_learning"),
        precision=32,
    )

    trainer.fit(model, DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=config.batch_size))

    # Create a dummy input for tracing
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, config.max_seq_length), dtype=torch.long)

    # Wrap the GPT2 model
    wrapped_model = GPT2Wrapper(model.model.gpt)
    
    # Export the trained model to TorchScript using tracing
    traced_model = torch.jit.trace(wrapped_model, dummy_input)
    traced_model.save("model.pt")
    print("Model exported to TorchScript format and saved as model.pt")

if __name__ == "__main__":
    config = ModelConfig(
        model_name="distilgpt2",
        num_tokens=10,
        num_tasks=5,
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=5,
        samples_per_task=50,
        max_seq_length=64,
        repo_name="DynamicFewShotMathGPT"
    )

    train_model(config)
