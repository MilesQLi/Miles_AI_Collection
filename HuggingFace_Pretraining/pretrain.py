from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer
)
import torch
import yaml
import argparse
import os

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_numeric_types(config):
    """Ensure numeric values in the config are properly typed."""
    
    # Define fields that should be integers
    integer_fields = [
        'max_seq_length', 'gradient_accumulation_steps', 'per_device_train_batch_size',
        'warmup_steps', 'max_steps', 'logging_steps', 'seed', 'max_samples', 'seq_len',
        'grad_accum_steps', 'batch_size_per_device', 'log_interval', 'epochs'
    ]
    
    # Define fields that should be floats
    float_fields = [
        'weight_decay', 'learning_rate'
    ]
    
    # Process all fields in a single loop
    for section in ['model', 'training', 'dataset', 'output']:
        if section in config:
            for field in integer_fields:
                if field in config[section]:
                    config[section][field] = int(config[section][field])
            for field in float_fields:
                if field in config[section]:
                    config[section][field] = float(config[section][field])
    
    return config

def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize the examples for pretraining."""
    # Tokenize the texts
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    return tokenized

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pretrain a language model')
    parser.add_argument('--config', type=str, default='default_config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure numeric values are properly typed
    config = ensure_numeric_types(config)
    
    # Extract model configuration
    model_config = config['model']
    model_path = model_config['name_or_path']
    max_seq_length = config['dataset']['seq_len']
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Load dataset configuration
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset_subset = dataset_config.get('subset', None)
    dataset_split = dataset_config['split']
    max_samples = dataset_config.get('max_samples', None)
    
    # Load and prepare the dataset
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Limit dataset size if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Apply tokenization function
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Extract training configuration
    training_config = config['training']
    
    # Create training arguments - FIXED: Added gradient clipping and mixed precision fixes
    training_args = TrainingArguments(
        gradient_accumulation_steps=training_config['grad_accum_steps'],
        per_device_train_batch_size=training_config['batch_size_per_device'],
        warmup_steps=training_config['warmup_steps'],
        learning_rate=training_config['learning_rate'],
        logging_steps=training_config['log_interval'],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=training_config['seed'],
        output_dir=config['output']['dir'],
        report_to="tensorboard",
        num_train_epochs=training_config['epochs'],
        # Fix for FP16 gradient unscaling issues
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        # Alternative: disable fp16 if issues persist
        # fp16=False,
        # bf16=True if torch.cuda.is_bf16_supported() else False,
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Create trainer - FIXED: Use processing_class for newer transformers versions
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        args=training_args,
    )
    
    # Set the processing_class (new way to set tokenizer)
    trainer.processing_class = tokenizer
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model
    trainer.save_model(config['output']['dir'])

if __name__ == "__main__":
    main()