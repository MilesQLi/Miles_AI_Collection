from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
import argparse
import os
from transformers import TrainerCallback
import json
from peft import LoraConfig, get_peft_model, TaskType

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

def formatting_prompts_func(examples, tokenizer, max_seq_length, question_column, answer_column):
    # Format the input prompt using apply_chat_template
    input_messages = [{"role": "user", "content": examples[question_column]}]
    input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True  # Include assistant's start token
    )
    
    # Format the full conversation using apply_chat_template
    full_messages = [
        {"role": "user", "content": examples[question_column]},
        {"role": "assistant", "content": examples[answer_column]}
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    #print("full_text",full_text)
    
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
    #print("max_seq_length",max_seq_length,type(max_seq_length))
    
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
    parser.add_argument('--gen_questions_file', type=str, default=None, help='Path to a file with generation questions (one per line)')
    # Removed gen_every_n_steps and gen_output_dir from argparse
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    gen_questions_file = args.gen_questions_file
    
    # Ensure numeric values are properly typed
    config = ensure_numeric_types(config)
    
    # Extract model configuration
    model_config = config['model']
    model_path = model_config['path']
    max_seq_length = model_config['max_seq_length']
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if enabled in config
    use_lora = model_config.get('use_lora', False)
    if use_lora:
        lora_config = model_config.get('lora_config', {})
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("LoRA applied successfully!")
    else:
        print("Training without LoRA (full fine-tuning)")
    
    # Load dataset configuration
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset_subset = dataset_config.get('subset', None)
    dataset_split = dataset_config['split']
    question_column = dataset_config.get('question_column', 'question')
    answer_column = dataset_config.get('answer_column', 'answer')
    
    # Load and prepare the dataset
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Apply formatting function with the tokenizer and max_seq_length
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer, max_seq_length, question_column, answer_column), 
        batched=False, 
        remove_columns=[question_column, answer_column]
    )
    
    # Extract training configuration
    training_config = config['training']
    # Get gen_every_n_steps from config, default to 100 if not present
    gen_every_n_steps = training_config.get('gen_every_n_steps', 20)
    # Set gen_output_dir as a subfolder of output_dir
    gen_output_dir = os.path.join(training_config['output_dir'], 'gen_results')
    if gen_questions_file:
        f = open(gen_questions_file, 'r')
        gen_questions = f.readlines()
        f.close()
        gen_questions = [q.strip() for q in gen_questions]
    else:
        gen_questions = None
    if gen_questions:
        os.makedirs(gen_output_dir, exist_ok=True)
    
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
        num_train_epochs=training_config['epochs'],
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

    # Custom callback for generation
    class GenerationCallback(TrainerCallback):
        def __init__(self, questions, tokenizer, output_dir, every_n_steps):
            self.questions = questions
            self.tokenizer = tokenizer
            self.output_dir = output_dir
            self.every_n_steps = every_n_steps
        def on_step_end(self, args, state, control, **kwargs):
            if self.questions and state.global_step % self.every_n_steps == 0 and state.global_step > 0:
                model = kwargs['model']
                model.eval()
                generations = []
                for q in self.questions:
                    messages = [
                        {"role": "user", "content": q}
                    ]

                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=256
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    generations.append({"question": q, "output": response})
                # Save to file
                out_path = os.path.join(self.output_dir, f"gen_step_{state.global_step}.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(generations, f, ensure_ascii=False, indent=2)
                print(f"[GenerationCallback] Saved generation results to {out_path}")

    # Add callback if needed
    if gen_questions:
        trainer.add_callback(GenerationCallback(
            questions=gen_questions,
            tokenizer=tokenizer,
            output_dir=gen_output_dir,
            every_n_steps=gen_every_n_steps
        ))

    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model
    if use_lora:
        # Create subfolder for LoRA weights
        lora_output_dir = os.path.join(training_config['output_dir'], 'lora_weights')
        os.makedirs(lora_output_dir, exist_ok=True)
        
        # Save LoRA weights in subfolder
        model.save_pretrained(lora_output_dir)
        print(f"LoRA weights saved to {lora_output_dir}")
        
        # Merge LoRA weights with base model and save complete model
        print("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        # Save the complete merged model in main output directory
        merged_model.save_pretrained(training_config['output_dir'])
        print(f"Complete merged model saved to {training_config['output_dir']}")
    else:
        # For full fine-tuning, use the trainer's save method
        trainer.save_model(training_config['output_dir'])

if __name__ == "__main__":
    main()