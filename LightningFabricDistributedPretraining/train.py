import torch
import argparse
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# Ensure Lightning is installed: pip install lightning
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger # Optional logger
from tqdm import tqdm
import warnings

# Suppress specific warnings if needed (e.g., from Hugging Face)
warnings.filterwarnings("ignore", message=".*but requires Rust dependencies.*")
warnings.filterwarnings("ignore", message=".*TorchScript is not supported.*")


# Define a simple PyTorch Dataset wrapper
class HFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def ensure_numeric_types(config):
    """Ensure numeric values in the config are properly parsed as numbers, not strings."""
    if isinstance(config, dict):
        return {key: ensure_numeric_types(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [ensure_numeric_types(item) for item in config]
    elif isinstance(config, str):
        # Try to convert string to float or int if possible
        try:
            return float(config) if '.' in config else int(config)
        except ValueError:
            return config
    return config

def load_dataset(tokenizer, config):
    """Loads and tokenizes a dataset."""
    dataset_config = config['dataset']
    seq_len = dataset_config['seq_len']
    max_samples = dataset_config['max_samples']
    
    print(f"Loading dataset (max_samples={max_samples}, seq_len={seq_len})...")
    
    # Load dataset from config
    raw_dataset = load_dataset(
        dataset_config['name'], 
        dataset_config['subset'],
        split=dataset_config['split'],
        trust_remote_code=dataset_config.get('trust_remote_code', True)
    )

    # Filter out empty or short texts
    raw_dataset = raw_dataset.filter(lambda example: example['text'] and len(example['text'].strip()) > 10)
    
    if len(raw_dataset) == 0:
        print("No valid samples found after filtering.")
        return None

    def tokenize_function(examples):
        # Tokenize texts with error handling
        tokenized_outputs = []
        for text in examples["text"]:
            try:
                output = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                    return_attention_mask=True,
                )
                tokenized_outputs.append(output)
            except Exception as e:
                print(f"Tokenization error: {e} | Text: {text[:100]}...")
                # Use pad tokens as fallback
                output = tokenizer(tokenizer.pad_token * seq_len, padding="max_length", max_length=seq_len, truncation=True, return_attention_mask=True)
                tokenized_outputs.append(output)

        # Combine results into batch format
        if not tokenized_outputs:
            return {}
            
        keys = tokenized_outputs[0].keys()
        return {key: [output[key] for output in tokenized_outputs] for key in keys}

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
        num_proc=max(os.cpu_count() // 2, 1)
    )

    # Prepare samples for CausalLM
    formatted_dataset = []
    print("Formatting dataset for Causal LM...")
    for example in tokenized_dataset:
        if "input_ids" not in example or "attention_mask" not in example:
            continue
            
        formatted_dataset.append({
            "input_ids": torch.tensor(example["input_ids"]),
            "attention_mask": torch.tensor(example["attention_mask"]),
            "labels": torch.tensor(example["input_ids"]).clone()  # For Causal LM, labels are the input_ids
        })

    print(f"Prepared {len(formatted_dataset)} samples.")
    return formatted_dataset

def main():
    parser = argparse.ArgumentParser(description="Lightning Fabric GPT-2 FSDP Training Example")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly parsed
    config = ensure_numeric_types(config)
    
    # Set torch matmul precision for better performance on Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # --- Initialize Fabric ---
    # Optional: Add a logger
    logger = TensorBoardLogger(root_dir=os.path.join(config['output']['dir'], "logs"))
    fabric = Fabric(
        accelerator=config['fabric']['accelerator'],
        strategy=config['fabric']['strategy'],
        devices=config['fabric']['devices'],
        precision=config['fabric']['precision'],
        num_nodes=config['fabric']['num_nodes'],
        loggers=logger,
    )
    fabric.launch() # Start the distributed processes/setup
    # --- End Fabric Initialization ---

    fabric.seed_everything(config['training']['seed'] + fabric.global_rank) # Set seed across all processes, add rank for variation

    if fabric.global_rank == 0:
        os.makedirs(config['output']['dir'], exist_ok=True) # Create output dir only on rank 0

    effective_batch_size = config['training']['batch_size_per_device'] * fabric.world_size * config['training']['grad_accum_steps']
    fabric.print(f"Effective batch size: {effective_batch_size}")
    fabric.print(f"Using Strategy: {type(fabric.strategy).__name__}")
    fabric.print(f"Using Precision: {fabric._precision}")

    fabric.print(f"Loading tokenizer: {config['model']['name_or_path']}")
    # Load tokenizer once, it's typically small and doesn't need distribution
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name_or_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        fabric.print("Added EOS token as PAD token.")

    # --- FSDP Aware Model Initialization ---
    # Use init_module for potentially large models with FSDP to avoid OOM on rank 0
    # empty_init=True can save memory by initializing on meta device first
    # Let Fabric decide based on strategy whether empty_init is beneficial
    with fabric.init_module():
        fabric.print(f"Loading model: {config['model']['name_or_path']} within fabric.init_module()")
        model = AutoModelForCausalLM.from_pretrained(config['model']['name_or_path'])
        model.config.pad_token_id = tokenizer.pad_token_id # Set pad token id
        # Optional: Apply activation checkpointing here if needed for very large models
        # if fabric.strategy.name == "fsdp":
        #     from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        #     from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        #     auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
        #     # You might need to pass this policy to the FSDPStrategy instance in Fabric initialization
        #     # Or apply it manually if Fabric doesn't expose easy wrapping policy config yet
        #     print("Note: Activation Checkpointing / Auto Wrapping Policy not explicitly applied in this example.")


    fabric.print("Loading and tokenizing dataset...")
    # Load the full dataset on each rank - Fabric will handle distribution
    train_data_list = load_dataset(tokenizer, config)
    
    if not train_data_list:
        fabric.print("Dataset is empty or failed to load. Exiting.")
        return
        
    train_dataset = HFDataset(train_data_list)
    
    def collate_fn(batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}
    
    # Setup DataLoader - Fabric will handle distribution and shuffling
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size_per_device'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() // fabric.world_size if fabric.world_size > 0 else 1),
        pin_memory=True
    )

    # Setup model with Fabric
    fabric.print("Setting up model with Fabric...")
    with fabric.init_module():
        model = AutoModelForCausalLM.from_pretrained(config['model']['name_or_path'])
        model.config.pad_token_id = tokenizer.pad_token_id

    # Setup optimizer and scheduler
    fabric.print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Calculate total training steps
    estimated_steps_per_epoch = len(train_dataset) // (config['training']['batch_size_per_device'] * fabric.world_size)
    total_training_steps = (estimated_steps_per_epoch // config['training']['grad_accum_steps']) * config['training']['epochs']
    
    if total_training_steps <= 0:
        fabric.print("Warning: Calculated total training steps is zero or negative. Using warmup steps as total.")
        total_training_steps = config['training']['warmup_steps'] * 2

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(config['training']['warmup_steps'], total_training_steps),
        num_training_steps=total_training_steps
    )

    # Setup model, optimizer, and dataloaders with Fabric
    fabric.print("Setting up model, optimizer, and dataloaders with Fabric...")
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    
    # Calculate steps per epoch
    actual_steps_per_epoch = len(train_dataloader)
    fabric.print(f"Steps per epoch: {actual_steps_per_epoch}")

    # Training loop
    fabric.print("Starting training...")
    total_processed_samples = 0
    current_grad_step = 0

    for epoch in range(config['training']['epochs']):
        fabric.print(f"--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        model.train()

        # Use tqdm only on rank 0
        iterable = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=(fabric.global_rank != 0))

        for batch_idx, batch in enumerate(iterable):
            is_accumulating = (batch_idx + 1) % config['training']['grad_accum_steps'] != 0

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss / config['training']['grad_accum_steps']

            # Backward pass
            fabric.backward(loss)

            # Step optimizer and scheduler after accumulation steps
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                if total_training_steps > 0:
                    lr_scheduler.step()
                current_grad_step += 1

                # Logging (on rank 0)
                if fabric.global_rank == 0 and current_grad_step % config['training']['log_interval'] == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    samples_in_step = config['training']['batch_size_per_device'] * fabric.world_size * config['training']['grad_accum_steps']
                    total_processed_samples += samples_in_step
                    
                    # Log metrics
                    fabric.log_dict({
                        "loss": loss.item() * config['training']['grad_accum_steps'],
                        "learning_rate": current_lr,
                        "epoch": epoch + (batch_idx / actual_steps_per_epoch),
                        "step": current_grad_step,
                    }, step=current_grad_step)

                    # Update tqdm description
                    iterable.set_postfix(
                        loss=f"{loss.item()*config['training']['grad_accum_steps']:.4f}",
                        lr=f"{current_lr:.2e}",
                        samples=f"{total_processed_samples}"
                    )

        # Handle any remaining gradients at the end of the epoch
        if len(train_dataloader) % config['training']['grad_accum_steps'] != 0:
            fabric.print("Performing final optimizer step for the epoch.")
            optimizer.step()
            optimizer.zero_grad()

    fabric.print("Training finished.")

    # Save the final model
    fabric.print(f"Saving final model checkpoint to {config['output']['dir']}")
    save_path = os.path.join(config['output']['dir'], f"final_model_epoch_{config['training']['epochs']}.pt")
    
    # Create state dictionary
    state = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler.state_dict(),
        "config": config,
        "epoch": config['training']['epochs'],
    }
    
    # Save model checkpoint
    fabric.save(save_path, state)
    
    # Save tokenizer on rank 0
    if fabric.global_rank == 0:
        tokenizer.save_pretrained(config['output']['dir'])
        fabric.print(f"Model and tokenizer saved to: {config['output']['dir']}")
        fabric.print("Training completed successfully.")

if __name__ == "__main__":
    main()