from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM
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
        'warmup_steps', 'max_steps', 'logging_steps', 'seed'
    ]
    
    # Define fields that should be floats
    float_fields = [
        'weight_decay', 'learning_rate'
    ]
    
    # Process all fields in a single loop
    for section in ['model', 'training']:
        if section in config:
            for field in integer_fields:
                if field in config[section]:
                    config[section][field] = int(config[section][field])
            for field in float_fields:
                if field in config[section]:
                    config[section][field] = float(config[section][field])
    
    return config

def formatting_prompts_func(examples, tokenizer, max_seq_length):
    # Format the input prompt using apply_chat_template
    input_messages = [{"role": "user", "content": examples['question']}]
    input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True  # Include assistant's start token
    )
    
    # Format the full conversation using apply_chat_template
    full_messages = [
        {"role": "user", "content": examples['question']},
        {"role": "assistant", "content": examples['answer']}
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize both texts
    input_tokenized = tokenizer(input_text, truncation=False, padding=False)
    full_tokenized = tokenizer(full_text, truncation=False, padding=False)
    
    # Get the length of the input tokens
    input_length = len(input_tokenized["input_ids"])
    
    # Create labels: -100 for input tokens, and real token IDs for output
    labels = [-100] * input_length  # Mask the input part
    
    # Append the remaining tokens from full_text
    if input_length < len(full_tokenized["input_ids"]):
        labels.extend(full_tokenized["input_ids"][input_length:])
    
    # Ensure all tensors have the same length
    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]
    print("max_seq_length",max_seq_length,type(max_seq_length))
    
    # Apply length constraints if needed
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    
    # Return the prepared tensors
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune a language model using SFT')
    parser.add_argument('--config', type=str, default='default_train_config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure numeric values are properly typed
    config = ensure_numeric_types(config)
    
    # Extract model configuration
    model_config = config['model']
    model_path = model_config['path']
    max_seq_length = model_config['max_seq_length']
    dtype = model_config['dtype']
    load_in_4bit = model_config['load_in_4bit']
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,
        torch_dtype=dtype if dtype else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset configuration
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset_subset = dataset_config.get('subset', None)
    dataset_split = dataset_config['split']
    
    # Load and prepare the dataset
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Apply formatting function with the tokenizer and max_seq_length
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer, max_seq_length), 
        batched=False, 
        remove_columns=["question", "answer"]
    )
    
    # Extract training configuration
    training_config = config['training']
    
    # Create training arguments - FIXED: Added gradient clipping and mixed precision fixes
    training_args = TrainingArguments(
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        warmup_steps=training_config['warmup_steps'],
        max_steps=training_config['max_steps'],
        learning_rate=training_config['learning_rate'],
        logging_steps=training_config['logging_steps'],
        weight_decay=training_config['weight_decay'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        seed=training_config['seed'],
        output_dir=training_config['output_dir'],
        report_to=training_config['report_to'],
        # Fix for FP16 gradient unscaling issues
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        # Alternative: disable fp16 if issues persist
        # fp16=False,
        # bf16=True if torch.cuda.is_bf16_supported() else False,
    )
    
    # Create trainer - FIXED: Use processing_class instead of tokenizer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        args=training_args,
    )
    
    # Set the processing_class (new way to set tokenizer)
    trainer.processing_class = tokenizer
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model
    trainer.save_model(training_config['output_dir'])

if __name__ == "__main__":
    main()