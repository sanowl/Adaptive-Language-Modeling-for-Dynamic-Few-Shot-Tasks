import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Dict, Optional
from dataclasses import dataclass
import math
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder
import secrets

# Load environment variables
load_dotenv()

# Configure logging
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
    hidden_size: int
    memory_size: int

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

class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class MAMLLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def adapt(self, loss: torch.Tensor, lr: float = 0.01) -> 'MAMLLayer':
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        return MAMLLayer(self.layer.in_features, self.layer.out_features).to(self.layer.weight.device)

class TaskAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.attention(query, key, value)[0]

class EpisodicMemory(nn.Module):
    def __init__(self, memory_size: int, embedding_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=1)

    def update(self, new_memory: torch.Tensor) -> None:
        attention_weights, _ = self.attention(new_memory.unsqueeze(0), self.memory.unsqueeze(0), self.memory.unsqueeze(0))
        self.memory = (1 - attention_weights) * self.memory + attention_weights * new_memory

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        attention_weights, _ = self.attention(query.unsqueeze(0), self.memory.unsqueeze(0), self.memory.unsqueeze(0))
        return (attention_weights * self.memory).sum(dim=0)

class DynamicPromptTuning(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gpt = AutoModelForCausalLM.from_pretrained(config.model_name)
        hidden_size = self.gpt.config.hidden_size
        self.prompt_embedding = nn.Parameter(torch.randn(config.num_tokens, hidden_size) / math.sqrt(hidden_size))
        self.task_layer = nn.Linear(hidden_size, hidden_size)
        self.task_attention = TaskAttention(hidden_size)
        nn.init.xavier_uniform_(self.task_layer.weight)
        nn.init.zeros_(self.task_layer.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, task_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.shape[0]

        prompt_embeddings = self.prompt_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.gpt.get_input_embeddings()(input_ids)

        combined_embeds = torch.cat((prompt_embeddings, inputs_embeds), dim=1)
        combined_embeds = self.task_attention(task_embedding.unsqueeze(0), combined_embeds, combined_embeds)

        if attention_mask is not None:
            prompt_attention = torch.ones(batch_size, self.config.num_tokens, device=attention_mask.device)
            attention_mask = torch.cat((prompt_attention, attention_mask), dim=1)

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
        self.prototypical_network = PrototypicalNetwork(hidden_size, hidden_size * 2, hidden_size)
        self.memory = DynamicMemory(config.num_tasks, config.num_tokens, hidden_size)
        self.maml_layer = MAMLLayer(hidden_size, hidden_size)
        self.episodic_memory = EpisodicMemory(config.memory_size, hidden_size)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        task_id = batch['task_id']
        if isinstance(task_id, torch.Tensor):
            task_id = task_id.item()
        task_embedding = self.memory.get(task_id)
        task_prototype = self.prototypical_network(task_embedding)
        adapted_task_embedding = self.maml_layer(task_prototype)
        
        episodic_memory = self.episodic_memory.retrieve(adapted_task_embedding)
        combined_embedding = adapted_task_embedding + episodic_memory
        
        outputs = self.model(input_ids=batch['input_ids'], 
                             attention_mask=batch['attention_mask'], 
                             labels=batch['labels'], 
                             task_embedding=combined_embedding)
        
        # Perform inner loop optimization
        adapted_layer = self.maml_layer.adapt(outputs.loss)
        adapted_task_embedding = adapted_layer(task_prototype)
        
        episodic_memory = self.episodic_memory.retrieve(adapted_task_embedding)
        combined_embedding = adapted_task_embedding + episodic_memory
        
        outputs = self.model(input_ids=batch['input_ids'], 
                             attention_mask=batch['attention_mask'], 
                             labels=batch['labels'], 
                             task_embedding=combined_embedding)
        
        # Update episodic memory
        self.episodic_memory.update(adapted_task_embedding.detach())
        
        return outputs.loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

        repo_id = f"{hf_username}/{self.config.repo_name}"

        api = HfApi()
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,
            exist_ok=True
        )

        model_path = "./hf_model"
        os.makedirs(model_path, exist_ok=True)
        
        self.model.gpt.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        torch.save(self.model.prompt_embedding, os.path.join(model_path, "prompt_embedding.pt"))
        torch.save(self.model.task_layer.state_dict(), os.path.join(model_path, "task_layer.pt"))
        torch.save(self.prototypical_network.state_dict(), os.path.join(model_path, "prototypical_network.pt"))
        torch.save(self.maml_layer.state_dict(), os.path.join(model_path, "maml_layer.pt"))
        torch.save(self.episodic_memory.state_dict(), os.path.join(model_path, "episodic_memory.pt"))
        
        upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            path_in_repo=".",
            token=hf_token,
            commit_message="Update model"
        )

        logger.info(f"Model pushed to Hugging Face Hub: {repo_url}")

class FewShotMathDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: AutoTokenizer, max_seq_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        input_text = f"Problem: {example['input']}\nSolution:"
        input_tokens = self.tokenizer(input_text, padding='max_length', truncation=True, 
                                      max_length=self.max_seq_length, return_tensors='pt')
        labels = self.tokenizer(example['output'], padding='max_length', truncation=True, 
                                max_length=self.max_seq_length, return_tensors='pt').input_ids
        input_tokens['labels'] = labels
        input_tokens = {k: v.squeeze(0) for k, v in input_tokens.items()}
        input_tokens['task_id'] = torch.tensor(idx // 10)  # Assign task_id based on idx for few-shot learning
        return input_tokens

def train_model(config: ModelConfig) -> None:
    pl.seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare few-shot dataset
    data = [
        {"input": "123 + 456", "output": "579"},
        {"input": "12 * 34", "output": "408"},
        {"input": "100 - 75", "output": "25"},
        {"input": "10 / 2", "output": "5"},
        # Add more examples here
    ]
    
    dataset = FewShotMathDataset(data, tokenizer, config.max_seq_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = DynamicMetaLearner(config)
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(monitor='val_loss')
        ],
        logger=TensorBoardLogger("logs", name="dynamic_few_shot_learning"),
        precision=32,
    )

    trainer.fit(model, DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), 
                DataLoader(val_dataset, batch_size=config.batch_size))

    save_directory = "./fine_tuned_model"
    os.makedirs(save_directory, exist_ok=True)

    tokenizer.save_pretrained(save_directory)
    model.model.gpt.save_pretrained(save_directory)
    torch.save(model.prototypical_network.state_dict(), os.path.join(save_directory, "prototypical_network.pt"))
    torch.save(model.maml_layer.state_dict(), os.path.join(save_directory, "maml_layer.pt"))
    torch.save(model.episodic_memory.state_dict(), os.path.join(save_directory, "episodic_memory.pt"))

    logger.info(f"Model and tokenizer saved to {save_directory}")

    # Export the model to TorchScript
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, config.max_seq_length), dtype=torch.long)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(os.path.join(save_directory, "model.pt"))
    logger.info("Model exported to TorchScript format and saved as model.pt")

def generate_math_problem(difficulty: str) -> Dict[str, str]:
    """Generate a random math problem based on difficulty."""
    if difficulty == "easy":
        a, b = secrets.SystemRandom().randint(1, 100), secrets.SystemRandom().randint(1, 100)
        op = secrets.choice(['+', '-'])
        problem = f"{a} {op} {b}"
        solution = str(eval(problem))
    elif difficulty == "medium":
        a, b = secrets.SystemRandom().randint(1, 20), secrets.SystemRandom().randint(1, 20)
        op = secrets.choice(['*', '//'])
        problem = f"{a} {op} {b}"
        solution = str(eval(problem))
    else:  # hard
        a, b, c = secrets.SystemRandom().randint(1, 20), secrets.SystemRandom().randint(1, 20), secrets.SystemRandom().randint(1, 20)
        op1, op2 = secrets.SystemRandom().choices(['+', '-', '*', '//'], k=2)
        problem = f"{a} {op1} {b} {op2} {c}"
        solution = str(eval(problem))
    
    return {"input": problem, "output": solution}

def create_dataset(num_examples: int, difficulties: List[str]) -> List[Dict[str, str]]:
    """Create a dataset with a specified number of examples and difficulties."""
    return [generate_math_problem(secrets.choice(difficulties)) for _ in range(num_examples)]

def evaluate_model(model: DynamicMetaLearner, test_dataset: Dataset, batch_size: int) -> float:
    """Evaluate the model on a test dataset."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            loss = model(batch)
            total_loss += loss.item()

    return total_loss / len(test_loader)

if __name__ == "__main__":
    config = ModelConfig(
        model_name="distilgpt2",
        num_tokens=10,
        num_tasks=5,
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=10,
        samples_per_task=50,
        max_seq_length=64,
        repo_name="AdvancedDynamicFewShotMathGPT",
        hidden_size=768,  # This should match the hidden size of the base model
        memory_size=100  # Size of the episodic memory
    )

    # Create a larger and more diverse dataset
    train_data = create_dataset(500, ["easy", "medium", "hard"])
    val_data = create_dataset(100, ["easy", "medium", "hard"])
    test_data = create_dataset(100, ["easy", "medium", "hard"])

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = FewShotMathDataset(train_data, tokenizer, config.max_seq_length)
    val_dataset = FewShotMathDataset(val_data, tokenizer, config.max_seq_length)
    test_dataset = FewShotMathDataset(test_data, tokenizer, config.max_seq_length)

    model = DynamicMetaLearner(config)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(monitor='val_loss', filename='best-checkpoint')
        ],
        logger=TensorBoardLogger("logs", name="dynamic_few_shot_learning"),
        precision=32,
    )

    trainer.fit(model, 
                DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=config.batch_size))

    # Evaluate the model on the test set
    test_loss = evaluate_model(model, test_dataset, config.batch_size)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Save the model
    save_directory = "./fine_tuned_model"
    os.makedirs(save_directory, exist_ok=True)

    tokenizer.save_pretrained(save_directory)
    model.model.gpt.save_pretrained(save_directory)
    torch.save(model.model.task_layer.state_dict(), os.path.join(save_directory, "task_layer.pt"))
    torch.save(model.prototypical_network.state_dict(), os.path.join(save_directory, "prototypical_network.pt"))
    torch.save(model.maml_layer.state_dict(), os.path.join(save_directory, "maml_layer.pt"))
    torch.save(model.episodic_memory.state_dict(), os.path.join(save_directory, "episodic_memory.pt"))

    logger.info(f"Model and tokenizer saved to {save_directory}")

    # Export the model to TorchScript
    dummy_input = {
        'input_ids': torch.randint(0, tokenizer.vocab_size, (1, config.max_seq_length), dtype=torch.long),
        'attention_mask': torch.ones((1, config.max_seq_length), dtype=torch.long),
        'labels': torch.randint(0, tokenizer.vocab_size, (1, config.max_seq_length), dtype=torch.long),
        'task_id': torch.tensor([0])
    }
    traced_model = torch.jit.trace(model, [dummy_input])
    traced_model.save(os.path.join(save_directory, "model.pt"))
    logger.info("Model exported to TorchScript format and saved as model.pt")

    # Push the model to Hugging Face Hub
    model.push_to_hub()

    logger.info("Training and evaluation completed successfully!")

