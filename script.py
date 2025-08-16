import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import json
from datetime import datetime   
from torch.optim import AdamW
from accelerate.utils import broadcast
from torch.utils.tensorboard import SummaryWriter


# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:

    
    # GPT2-Medium configuration (from scratch) - 345M
    medium_config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=4096,
        activation_function="gelu_new", # changed from gelu
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )
    
    # Training params
    batch_size = 32
    max_seq_length = 512
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Training control
    num_epochs = 3
    learning_rate = 1e-5
    min_token_length = 30
    checkpoint_steps = 5000
    checkpoint_dir = "checkpoints"
    load_checkpoint = False

# Initialize Accelerator
accelerator = Accelerator()
os.makedirs("batchwise_plots", exist_ok=True)

class Dataset_class(IterableDataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("Skylion007/openwebtext", 
                                  split=split, 
                                  streaming=False)
        self.filtered_dataset = self.dataset.filter(
            lambda x: len(x["text"].split()) >= Config.min_token_length
        )

    def __iter__(self):
        return iter(self.filtered_dataset)


    def serialize(self):
        """Converts the current state of the object into a dictionary."""
        return {
            'mean': self.mean,
            'var': self.var,
            'counter': self.counter,
            'history': self.history
        }
    
    @classmethod
    def deserialize(cls, state):
        """Creates a new ThresholdEMA object from a saved state dictionary."""
        obj = cls(state['mean'], state['var'])
        obj.counter = state.get('counter', 0)
        obj.history = state.get('history', [])
        return obj


def get_latest_checkpoint():
    """Get the most recent checkpoint directory"""
    if not os.path.exists(Config.checkpoint_dir):
        return None
        
    dirs = [f for f in os.scandir(Config.checkpoint_dir) if f.is_dir()]
    if not dirs:
        return None
        
    # Sort directories by creation time (oldest first)
    dirs.sort(key=lambda x: x.stat().st_ctime)
    
    # Get the most recent directory (last in sorted list)
    return dirs[-1].path

def save_checkpoint(model, optimizer, epoch, global_step, batch_step, threshold_ema):
    """Save checkpoint with timestamp-based directory"""
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Create unique directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(Config.checkpoint_dir, f"checkpoint_{timestamp}")
    
    # Create directory and save state
    os.makedirs(checkpoint_path, exist_ok=True)
        
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_step': batch_step,
        'threshold_ema': threshold_ema
    }, os.path.join(checkpoint_path, "training_state.pth"))
    
    logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, small_model, small_optimizer):
    """Load checkpoint from directory"""
    state = torch.load(os.path.join(checkpoint_path, "training_state.pth"))
    
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    small_model.load_state_dict(state['small_model_state_dict'])
    small_optimizer.load_state_dict(state['small_optimizer_state_dict'])
    threshold_ema = ThresholdEMA.deserialize(state['threshold_ema'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return state['epoch'], state['global_step'], state['batch_step'], threshold_ema


def train_model_baseline(model, tokenizer, dataset, use_weighting=True, model_name="model"):
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        tokenized_texts = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=Config.max_seq_length
            ).to(accelerator.device)
        return tokenized_texts

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer,
        DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, collate_fn=collate_fn)
    )

    threshold_ema = None
    losses = []
    start_epoch = 0
    batch_step = 0
    global_step = 0

    if Config.load_checkpoint:
        latest_checkpoint = get_latest_checkpoint()
        if latest_checkpoint:
            start_epoch, global_step, batch_step, threshold_ema = load_checkpoint(latest_checkpoint, model, optimizer, small_model, small_optimizer)
            logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    large_loss_idx_value = 0
    small_update_step = 0
    
    for epoch in range(start_epoch, Config.num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - {model_name}", 
                          disable=not accelerator.is_local_main_process)
        
        if epoch != start_epoch: # Reset batch step if resuming from checkpoint
            batch_step = 0
        batch_idx = 0
        logger.info("Starting epoch")
        
        for batch in progress_bar:
            print(batch_idx)
            if batch_idx < batch_step:
                continue  # Skip steps if resuming from checkpoint

            inputs = batch
            # Forward pass
            logger.info("Training baseline model")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss #----------------> changed
            
            ### Changed
            
            all_losses_sum = accelerator.gather(loss)
        
            ### Changed
            if accelerator.is_main_process:
                if len(all_losses_sum) > 0: # Avoid division by zero
                    print('effective batch size',len(all_losses_sum))
                    global_avg_loss = all_losses_sum.mean().item()
                    losses.append(global_avg_loss)
                    writer.add_scalar("Baseline Loss", global_avg_loss, large_loss_idx_value)
            large_loss_idx_value += 1

         
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            

            if global_step % Config.checkpoint_steps == 0:
                save_checkpoint(model, optimizer, epoch, global_step, batch_idx, threshold_ema)
                       
            global_step += 1
            batch_step += 1
            batch_idx += 1

        if accelerator.is_main_process:
            for name, param in model.named_parameters():
                writer.add_histogram(f"Baseline weights/{name}", param, global_step=epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Baseline gradients/{name}", param.grad, global_step=epoch)
        
    return losses



def evaluate_model(model, tokenizer, split="test"):
    model.eval()
    dataset = Dataset_class(split)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size)
    dataloader = accelerator.prepare(dataloader)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", 
                        disable=not accelerator.is_local_main_process):
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=Config.max_seq_length
            ).to(accelerator.device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["attention_mask"].sum().item()
            total_tokens += inputs["attention_mask"].sum().item()
    
    total_loss = accelerator.reduce(torch.tensor(total_loss, device=accelerator.device))
    total_tokens = accelerator.reduce(torch.tensor(total_tokens, device=accelerator.device))
    
    avg_loss = total_loss.item() / total_tokens.item()
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"Evaluation Results - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return {"loss": avg_loss, "perplexity": perplexity}

def main():
    # Initialize models
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    baseline_model = GPT2LMHeadModel(Config.medium_config)
    
    # Load dataset
    train_dataset = Dataset_class("train")
    
    
    logger.info("Starting training for baseline model")
    baseline_losses = train_model_baseline(
        baseline_model,
        tokenizer,
        train_dataset,
        use_weighting=False,
        model_name="baseline"
    )

    if accelerator.is_main_process:
        with open("baseline_losses.json", "w") as f:
            json.dump({
                "losses": baseline_losses
            }, f)
        logger.info("Saved baseline losses to baseline_losses.json")
    
    # Plot of losses
    if accelerator.is_main_process:
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        plt.subplot(1, 2, 1)
        plt.plot(range(len(baseline_losses)), baseline_losses, label="Baseline Model", color="orange")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss vs. Steps")
        plt.legend()
        plt.grid(True)
        
        # Calculate and plot perplexity
        perplexities = [torch.exp(torch.tensor(loss)).item() for loss in baseline_losses]
        plt.subplot(1, 2, 2)
        plt.plot(range(len(perplexities)), perplexities, label="Baseline Model", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Perplexity")
        plt.title("Perplexity vs. Steps")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_metrics.png")  # Save the plot as an image
        plt.close()
    
    # Evaluation
    logger.info("Evaluating models")
    # baseline_metrics = evaluate_model(baseline_model, tokenizer)
    
    # Final results
    if accelerator.is_main_process:
        logger.info("\nFinal Results:")
        # logger.info(f"Baseline Model - Loss: {baseline_metrics['loss']:.4f}, "
        #            f"Perplexity: {baseline_metrics['perplexity']:.2f}")
        
        # Save final models
        torch.save(accelerator.unwrap_model(baseline_model).state_dict(), "baseline_model.pth")
        logger.info("Saved final models")

if __name__ == "__main__":
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter()
    main()
    if accelerator.is_main_process:
        writer.flush()
        writer.close()
