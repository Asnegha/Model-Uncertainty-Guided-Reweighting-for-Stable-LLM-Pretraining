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
from sklearn.mixture import GaussianMixture
import math
from tqdm import tqdm


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

import yaml
import torch
from transformers import GPT2Config

class Config:
    def __init__(self, config_path="args.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.dataset = config['dataset']
        
        # Model specs
        self.small_model_name = config['model']['small_model_name']
        
        # GPT2-Medium configuration (from scratch)
        medium_config = config['model']['medium_model']
        self.medium_config = GPT2Config(
            vocab_size=int(medium_config['vocab_size']),
            n_positions=int(medium_config['n_positions']),
            n_embd=int(medium_config['n_embd']),
            n_layer=int(medium_config['n_layer']),
            n_head=int(medium_config['n_head']),
            n_inner=int(medium_config['n_inner']),
            activation_function=medium_config['activation_function'],
            resid_pdrop=float(medium_config['resid_pdrop']),
            embd_pdrop=float(medium_config['embd_pdrop']),
            attn_pdrop=float(medium_config['attn_pdrop']),
            layer_norm_epsilon=float(medium_config['layer_norm_epsilon']),
            initializer_range=float(medium_config['initializer_range']),
        )
        
        # Training params
        training = config['training']
        self.batch_size = int(training['batch_size'])
        self.max_seq_length = int(training['max_seq_length'])
        self.num_workers = int(training['num_workers'])
        self.device = torch.device(training['device'] if torch.cuda.is_available() else "cpu")
        
        # small model gradient update
        self.k = int(training['k'])
        
        # EMA parameters
        self.ema_alpha = float(training['ema_alpha'])
        
        # Weighting parameters
        self.alpha = float(training['alpha'])
        self.beta = float(training['beta'])
        
        # Training control
        self.num_epochs = int(training['num_epochs'])
        self.learning_rate = float(training['learning_rate'])
        self.min_token_length = int(training['min_token_length'])
        self.plot_every_n_batches = int(training['plot_every_n_batches'])
        self.checkpoint_steps = int(training['checkpoint_steps'])
        self.checkpoint_dir = training['checkpoint_dir']
        self.load_checkpoint = training['load_checkpoint']

# Create a config instance
config = Config()

# Initialize Accelerator
accelerator = Accelerator()
print(torch.cuda.device_count())
os.makedirs("batchwise_plots", exist_ok=True)

class Dataset_class(IterableDataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset(config.dataset, 
                                  split=split, 
                                  streaming=False, trust_remote_code=True)
        self.filtered_dataset = self.dataset.filter(
            lambda x: len(x["text"].split()) >= config.min_token_length
        )

    def __iter__(self):
        return iter(self.filtered_dataset)
    
# Weight calculation function
# def calculate_weight(mean, var, global_mean, global_var, hard_cluster_mean, noisy_cluster_mean, cluster_label):
#     if cluster_label == 0:  # Hard cluster
#         ref_mean, ref_var = hard_cluster_mean
#     else:  # Noisy cluster
#         ref_mean, ref_var = noisy_cluster_mean

#     d1 = np.linalg.norm(np.array([global_var, global_mean]) - np.array([global_var, 0]))
#     d2 = np.linalg.norm(np.array([global_var, global_mean]) - np.array([mean, var]))
#     d3 = np.linalg.norm(np.array([ref_var, ref_mean]) - np.array([mean, var]))
#     d = np.linalg.norm(np.array([global_var, 0]) - np.array([mean, var]))

#     numerator = d1**2 + d2**2 - d**2
#     denominator = 2 * d1 * d2
#     angle = math.acos(numerator / denominator)

#     if cluster_label == 0:  # Hard cluster
#         weight = np.exp(angle / 180) + d3
#     else:  # Noisy cluster
#         weight = np.exp(angle / 180) - d3

#     return weight

def calculate_weight(mean, var, global_mean, global_var, ref_mean, ref_var, cluster_type):
    d1 = np.linalg.norm(np.array([global_var, global_mean]) - np.array([global_var, 0]))
    d2 = np.linalg.norm(np.array([global_var, global_mean]) - np.array([mean, var]))
    d3 = np.linalg.norm(np.array([ref_var, ref_mean]) - np.array([mean, var]))
    d = np.linalg.norm(np.array([global_var, 0]) - np.array([mean, var]))

    numerator = d1**2 + d2**2 - d**2
    denominator = 2 * d1 * d2
    angle = math.acos(numerator / denominator)

    if cluster_type == "hard":  # Hard cluster
        weight = np.exp(angle / 180) + d3
    elif cluster_type == "noisy":  # Noisy cluster
        weight = np.exp(angle / 180) - d3
    else:
        weight = 1

    return weight

def plot_batch_metrics(batch_idx, means, vars, weights, ema_mean, ema_var, model_name):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(means, vars, c=weights, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Sample Weight')
    plt.axvline(ema_mean, color='r', linestyle='--', label=f'EMA Mean ({ema_mean:.2f})')
    plt.axhline(ema_var, color='g', linestyle='--', label=f'EMA Var ({ema_var:.2f})')
    plt.xlabel('Mean Loss')
    plt.ylabel('Variance')
    plt.title(f'Batch {batch_idx} - {model_name}')
    plt.legend()
    plt.savefig(f"batchwise_plots/{model_name}_batch_{batch_idx}.png")
    plt.close()

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

def save_checkpoint(model, optimizer, small_model, small_optimizer, epoch, global_step, batch_step, threshold_ema):
    """Save checkpoint with timestamp-based directory"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create unique directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{timestamp}")
    
    # Create directory and save state
    os.makedirs(checkpoint_path, exist_ok=True)

    temp_small_model_state_dict = None
    temp_small_optimizer_state_dict = None

    if small_model is not None:
        temp_small_model_state_dict = small_model.state_dict()
        temp_small_optimizer_state_dict = small_optimizer.state_dict()
    
    # if threshold_ema is not None:
    #     threshold_ema = threshold_ema.serialize()
    # else:
    #     threshold_ema = ThresholdEMA(0, 0, 0).serialize()
        
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'small_model_state_dict': temp_small_model_state_dict,
        'small_optimizer_state_dict': temp_small_optimizer_state_dict,
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
    # threshold_ema = ThresholdEMA.deserialize(state['threshold_ema'])
    threshold_ema = 0
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return state['epoch'], state['global_step'], state['batch_step'], threshold_ema


def train_model_weighted(model, small_model, tokenizer, dataset, use_weighting=True, model_name="model"):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    small_optimizer = AdamW(small_model.parameters(), lr=config.learning_rate)
    
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        tokenized_texts = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length
            ).to(accelerator.device)
        return tokenized_texts

    model, optimizer, small_model, small_optimizer, dataloader = accelerator.prepare(
        model, optimizer, small_model, small_optimizer,
        # DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers)
        DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn)
    )
    
    # small_model.to(accelerator.device)
    # small_model.eval()
    
    threshold_ema = None
    losses = []
    start_epoch = 0
    batch_step = 0
    global_step = 0

    if config.load_checkpoint:
        latest_checkpoint = get_latest_checkpoint()
        if latest_checkpoint:
            start_epoch, global_step, batch_step, threshold_ema = load_checkpoint(latest_checkpoint, model, optimizer, small_model, small_optimizer)
            logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    small_loss_idx_value = 0
    large_loss_idx_value = 0
    small_update_step = 0
    
    for epoch in tqdm(range(start_epoch, config.num_epochs)):
        model.train()
        small_model.train() 
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - {model_name}", 
                          disable=not accelerator.is_local_main_process)
        
        if epoch != start_epoch: # Reset batch step if resuming from checkpoint
            batch_step = 0
        batch_idx = 0
        logger.info("Starting epoch")
        
        num_bins = 49  # one less, because we’ll manually add the last open-ended bin
        hist_bins = np.concatenate([
            np.linspace(0, 50, num_bins + 1),  # 0 to 10
            [np.inf]  # final bin for (10, ∞)
        ])
        hist_counts = np.zeros(len(hist_bins) - 1)  # 10 bins total


        for batch in progress_bar:
            small_update_step += 1
            # print(batch_idx)
            if batch_idx < batch_step:
                continue  # Skip steps if resuming from checkpoint

            inputs = batch
            logger.info("Training small model")
            # small_optimizer.zero_grad()
            small_outputs = small_model(**inputs, labels=inputs["input_ids"])
            small_logits = small_outputs.logits
            
            logger.info("Calculating losses and weights")
            # Calculate losses and weights
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits = small_logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            mask = inputs["attention_mask"][:, 1:].float()
            
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                    shift_labels.view(-1)) # batch_size X seq_len (tokens) -> one loss per each token in the sequence for all batch 
            per_token_loss = per_token_loss.view(shift_labels.shape) * mask
            
            valid_counts = mask.sum(dim=1)
            sample_means = (per_token_loss.sum(dim=1) / valid_counts) # batch_size X 1 => average among the sequence
            
            
            
            small_loss = sample_means.mean() # one real value

            # changed
            # Multi GPU gathering
            
            all_small_losses_sum = accelerator.gather(small_loss)
            
            ### Changed
            if accelerator.is_main_process:
                global_small_avg_loss = all_small_losses_sum.mean().item()
                writer.add_scalar("Small Model Loss", global_small_avg_loss, small_loss_idx_value)
            small_loss_idx_value += 1
            
            ### 
            
            accelerator.backward(small_loss)
            small_optimizer.step()  
            small_optimizer.zero_grad()
    
            # Gather metrics across GPUs
            with torch.no_grad():
                sample_means_cpu = sample_means.cpu().numpy()
                sample_vars_cpu = [(per_token_loss[i] - sample_means_cpu[i]).pow(2).sum().item() / 
                         valid_counts[i].item() for i in range(len(sample_means_cpu))]
            
            # print('sample_means_cpu', sample_means, sample_means_cpu) #----------------> changed
            all_means = accelerator.gather(torch.tensor(sample_means_cpu, device=accelerator.device))
            all_vars = accelerator.gather(torch.tensor(sample_vars_cpu, device=accelerator.device))
            global_batch_mean = all_means.mean()
            global_batch_var = all_vars.mean()
            
            gmm = GaussianMixture(n_components=3, random_state=42)
            features = np.array(list(zip(all_means.cpu(), all_vars.cpu())))
            gmm.fit(features)
            cluster_means = gmm.means_
            print( cluster_means)

            # Create a mapping from original cluster labels to sorted positions
            # sorted_indices = cluster_means[:, 0].argsort()  # Get the sort order
            # sorted_means = cluster_means[sorted_indices]
            
            sum_components = cluster_means[:, 0] + cluster_means[:, 1]  # Mean + Variance
            sorted_indices = np.argsort(sum_components)  # Get indices that would sort by sum
            sorted_means = cluster_means[sorted_indices]
            
            
            label_to_type = {
                sorted_indices[0]: "easy",
                sorted_indices[1]: "hard",
                sorted_indices[2]: "noisy"
            }
            
            # Create reference means dictionary
            ref_means = {
                "easy": sorted_means[0],
                "hard": sorted_means[1],
                "noisy": sorted_means[2]
            }
            
            if accelerator.is_main_process:
                global_cluster_labels = gmm.predict(features)
                
                plt.figure(figsize=(10, 8))

                # Plot each cluster with different colors
                colors = {
                    "easy": 'green',
                    "hard": 'red',
                    "noisy": 'blue'
                }

                # Plot all points with their cluster colors
                for mean, var, original_label in zip(all_means, all_vars, global_cluster_labels):
                    mean = torch.tensor(mean) if not torch.is_tensor(mean) else mean.cpu()
                    var = torch.tensor(var) if not torch.is_tensor(var) else var.cpu()
                
                    cluster_type = label_to_type[original_label]
                    plt.scatter(mean, var, c=colors[cluster_type], alpha=0.6, label=cluster_type)

                # Plot cluster centers
                for cluster_type, (mean, var) in ref_means.items():
                    plt.scatter(mean, var, c='black', marker='x', s=200, linewidths=3)
                    plt.text(mean, var, f'{cluster_type} center', fontsize=12)

                # Remove duplicate labels
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

                # Add labels and title
                plt.xlabel('Mean')
                plt.ylabel('Variance')
                plt.title('Batch-wise Clustering of Samples')
                plt.grid(True)

                # Save the plot
                plt.savefig('cluster_plot/batch_cluster_plot'+ str(batch_idx) + '.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            
            
            
                      
            features_local = np.array(list(zip(sample_means_cpu, sample_vars_cpu)))
            cluster_labels = gmm.predict(features_local)
              # Sort means

            # Create a dictionary to map original labels to cluster types
            

            
                
            weights = []
            for mean, var, original_label in zip(sample_means_cpu, sample_vars_cpu, cluster_labels):
                cluster_type = label_to_type[original_label]
                ref_mean, ref_var = ref_means[cluster_type]
                
                mean_t = torch.tensor(mean) if not torch.is_tensor(mean) else mean.cpu()
                var_t = torch.tensor(var) if not torch.is_tensor(var) else var.cpu()
                global_mean = torch.tensor(global_batch_mean) if not torch.is_tensor(global_batch_mean) else global_batch_mean.cpu()
                global_var = torch.tensor(global_batch_var) if not torch.is_tensor(global_batch_var) else global_batch_var.cpu()
                
                
                weight = calculate_weight(
                    mean_t, var_t,
                    global_mean, global_var,
                    ref_mean, ref_var,
                    cluster_type
                )
                weights.append(weight)

            weights = torch.tensor(weights, device=accelerator.device, dtype=torch.float32)
            weights = torch.nn.functional.softmax(weights)
            
            # print(all_means.shape, all_vars.shape) #----------------> changed

            # Initialize ThresholdEMA
            logger.info(f"{accelerator.is_main_process}")
            
            
            # # Calculate weights
            # weights = []
            # for mean, var in zip(sample_means_cpu, sample_vars_cpu):
            #     # Convert to tensors if they aren't already
            #     mean_t = torch.tensor(mean) if not torch.is_tensor(mean) else mean.cpu()
            #     var_t = torch.tensor(var) if not torch.is_tensor(var) else var.cpu()
            #     global_mean_t = torch.tensor(global_batch_mean) if not torch.is_tensor(global_batch_mean) else global_batch_mean.cpu()
            #     global_var_t = torch.tensor(global_batch_var) if not torch.is_tensor(global_batch_var) else global_batch_var.cpu()
                
            #     if mean_t <= global_mean_t and var_t >= global_var_t:
            #         d1 = torch.norm(torch.tensor([global_var_t, global_mean_t]) - torch.tensor([global_var_t, 0]))
            #         d2 = torch.norm(torch.tensor([global_var_t, global_mean_t]) - torch.tensor([mean_t, var_t]))
            #         d = torch.norm(torch.tensor([global_var_t, 0]) - torch.tensor([mean_t, var_t]))
            #         numerator = d1**2 + d2**2 - d**2
            #         denominator = 2 * d1 * d2
            #         angle = torch.acos(numerator / denominator)
            #         weight = torch.exp(angle/180) + (d2)
                    
            #     elif mean_t > global_mean_t and var_t > global_var_t:
            #         d1 = torch.norm(torch.tensor([global_var_t, global_mean_t]) - torch.tensor([global_var_t, 0]))
            #         d2 = torch.norm(torch.tensor([global_var_t, global_mean_t]) - torch.tensor([mean_t, var_t]))
            #         d = torch.norm(torch.tensor([global_var_t, 0]) - torch.tensor([mean_t, var_t]))
            #         numerator = d1**2 + d2**2 - d**2
            #         denominator = 2 * d1 * d2
            #         angle = torch.acos(numerator / denominator)
            #         weight = torch.exp(angle/180) - (d2)
            #     else:
            #         weight = 1.0
            #     weights.append(weight)
            # weights = torch.tensor(weights, device=accelerator.device, dtype=torch.float32)
            # weights = torch.nn.functional.softmax(weights)
            # Gather weights across GPUs
            all_weights = accelerator.gather(weights)
            # print(f"Weights: {all_weights}") #----------------> changed

            # Update histogram counts with gathered weights
            # changed
            if accelerator.is_main_process:
                all_weights_np = all_weights.detach().cpu().numpy()
                batch_hist, _ = np.histogram(all_weights_np, bins=hist_bins)
                hist_counts += batch_hist
            # print('weights', weights.shape, weights) #----------------> changed
            # Forward pass
            logger.info("Training medium model")
            outputs = model(**inputs, labels=inputs["input_ids"])
            # loss = outputs.loss #----------------> changed
            
            ### Changed
            medium_logits = outputs.logits
            loss_fct_medium = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits_medium = medium_logits[:, :-1, :].contiguous()
            shift_labels_medium = inputs["input_ids"][:, 1:].contiguous()
            mask_medium = inputs["attention_mask"][:, 1:].float()
            per_token_loss_medium = loss_fct_medium(shift_logits_medium.view(-1, shift_logits_medium.size(-1)), 
                                shift_labels_medium.view(-1))
            loss = per_token_loss_medium.view(shift_labels_medium.shape) * mask_medium
            valid_counts_medium = mask_medium.sum(dim=1)
            loss = (loss.sum(dim=1) / valid_counts_medium)
            # print('pre sample loss medoum model',loss.shape)
            
            ###
        
            ### Changed

            all_losses_sum = accelerator.gather(loss)
            ### Changed
            if accelerator.is_main_process:
                global_avg_loss = all_losses_sum.mean().item()
                losses.append(global_avg_loss)
                writer.add_scalar("Medium Model Loss", global_avg_loss, large_loss_idx_value)
            large_loss_idx_value += 1

            ###
            
            # Apply weighting
            if use_weighting:
                # print('Weighting the loss', loss.shape, weights.shape) #----------------> changed
                loss = (loss * weights).mean()    
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            # losses.append(loss.item()) #----------------> changed
            if accelerator.is_main_process:
                logger.info(f"Step {global_step}: Loss={loss.item():.4f}")
            
            if global_step % config.checkpoint_steps == 0:
                save_checkpoint(model, optimizer, small_model, small_optimizer, epoch, global_step, batch_idx, threshold_ema)
            batch_weights = accelerator.gather(weights)
            # Plotting
            if accelerator.is_main_process:
                if global_step % config.plot_every_n_batches == 0 and accelerator.is_main_process:
                    plot_batch_metrics(
                        global_step,
                        all_means.cpu(),
                        all_vars.cpu(),
                        batch_weights.cpu().numpy(),
                        global_batch_mean.cpu(),
                        global_batch_var.cpu(),
                        model_name
                    )
            
            global_step += 1
            batch_step += 1
            batch_idx += 1
            
        if accelerator.is_main_process:
            if writer:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Medium weights/{name}", param, global_step=epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Medium gradients/{name}", param.grad, global_step=epoch)

        if accelerator.is_main_process: 
            if writer:
                for name, param in small_model.named_parameters():
                    writer.add_histogram(f"Small weights/{name}", param, global_step=epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Small gradients/{name}", param.grad, global_step=epoch)
        
        if accelerator.is_main_process:
            bin_edges = hist_bins.copy()
            bin_edges[-1] = bin_edges[-2] + 10  # You can change this width if needed

            bin_widths = np.diff(bin_edges)
            bin_centers = bin_edges[:-1] + bin_widths / 2

            xtick_labels = [f"{int(hist_bins[i])}-{int(hist_bins[i+1])}" if not np.isinf(hist_bins[i+1])
                    else f">{int(hist_bins[i])}" for i in range(len(hist_counts))]

            plt.figure(figsize=(10, 5))
            plt.bar(bin_centers, hist_counts, width=bin_widths, edgecolor='black', color='skyblue')
            plt.xticks(bin_centers, xtick_labels, rotation=45)
            plt.xlabel("Weight Bin")
            plt.ylabel("Frequency")
            plt.title(f"Weight Distribution Across All Batches - {epoch+1}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{epoch+1}_weight_distribution.png")
            plt.close()

    return losses, threshold_ema.history if threshold_ema else [], model, small_model

def evaluate_model(model, tokenizer, split="test"):
    model.eval()
    dataset = Dataset_class(split)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
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
                max_length=config.max_seq_length
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
    tokenizer = GPT2Tokenizer.from_pretrained(config.small_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    small_model = GPT2LMHeadModel.from_pretrained(config.small_model_name)
    medium_model = GPT2LMHeadModel(config.medium_config)
    baseline_model = GPT2LMHeadModel(config.medium_config)
    
    # Load dataset
    train_dataset = Dataset_class("train")
    
    # Training
    logger.info("Starting training for weighted model")
    weighted_losses, ema_history, medium_model, small_model = train_model_weighted(
        medium_model,
        small_model,
        tokenizer,
        train_dataset,
        use_weighting=True,
        model_name="weighted"
    )
    if accelerator.is_main_process:
        with open("weighted_losses.json", "w") as f:
            json.dump({
                "losses": weighted_losses
            }, f)
        logger.info("Saved weighted losses to weighted_losses.json")

    
    # Plot of losses
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(medium_model).state_dict(), "weighted_medium_model.pth")
        torch.save(accelerator.unwrap_model(small_model).state_dict(), "small_model.pth")
        logger.info("Saved final models")
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        plt.subplot(1, 2, 1)
        plt.plot(range(len(weighted_losses)), weighted_losses, label="Baseline Model", color="orange")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss vs. Steps")
        plt.legend()
        plt.grid(True)
        
        # Calculate and plot perplexity
        perplexities = [torch.exp(torch.tensor(loss)).item() for loss in weighted_losses]
        plt.subplot(1, 2, 2)
        plt.plot(range(len(perplexities)), perplexities, label="Baseline Model", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Perplexity")
        plt.title("Perplexity vs. Steps")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_metrics_weighted.png")  # Save the plot as an image
        plt.close()
    
    # # Evaluation
    logger.info("Evaluating models")
    weighted_metrics = evaluate_model(medium_model, tokenizer)
    baseline_metrics = evaluate_model(baseline_model, tokenizer)
    
    # # Final results
    if accelerator.is_main_process:
         logger.info("\nFinal Results:")
         logger.info(f"Weighted Model - Loss: {weighted_metrics['loss']:.4f}, "
                    f"Perplexity: {weighted_metrics['perplexity']:.2f}")

        
        # Save final models
        

if __name__ == "__main__":
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter()
    main()
    if accelerator.is_main_process:
        writer.flush()
        writer.close()
