# HuggingFace Model Pretraining Script

A flexible and configurable script for pretraining language models using the HuggingFace Transformers library. This script supports causal language modeling (CLM) pretraining on custom datasets.

## Features

- Configurable model architecture and training parameters via YAML
- Support for different datasets from HuggingFace Hub
- Mixed precision training (FP16) for improved efficiency
- TensorBoard logging support
- Automatic handling of tokenization and data collation
- Configurable dataset size limits and sequence lengths

## Requirements

```python
datasets>=2.0.0
transformers>=4.0.0
torch>=1.8.0
pyyaml
tensorboard
accelerate
```

## Setup

Before running the training script, you should configure the Accelerate environment. This sets up the distributed training arguments and hardware configuration. Run the following command and follow the prompts:

```bash
accelerate config
```

This only needs to be done once per environment or whenever you want to change your distributed training setup.

## Configuration

The script uses a YAML configuration file to specify model, dataset, and training parameters. Here's an example configuration structure:

```yaml
model:
  name_or_path: "gpt2"  # or any other model from HuggingFace Hub
  max_seq_length: 512

dataset:
  name: "your_dataset_name"  # Dataset name from HuggingFace Hub
  subset: "subset_name"      # Optional dataset subset
  split: "train"            # Dataset split to use
  max_samples: 1000000      # Optional: limit number of training samples
  seq_len: 512             # Sequence length for tokenization

training:
  batch_size_per_device: 8
  grad_accum_steps: 4
  learning_rate: 5e-5
  warmup_steps: 1000
  epochs: 3
  seed: 42
  log_interval: 100

output:
  dir: "output_directory"   # Directory to save the model
```

## Usage

1. Create a configuration YAML file with your desired settings.

2. Run the pretraining script:
```bash
accelerate launch pretrain.py --config default_pretrain_config.yaml
```

## Training Process

The script performs the following steps:

1. Loads and validates the configuration from the YAML file
2. Initializes the model and tokenizer from HuggingFace Hub
3. Loads and preprocesses the dataset
4. Sets up the training arguments and data collator
5. Trains the model using HuggingFace's `Trainer`
6. Saves the trained model to the specified output directory

## Features in Detail

### Automatic Type Conversion
The script automatically converts configuration values to appropriate numeric types (integer or float) for relevant parameters.

### Tokenization
- Handles tokenization with proper padding and truncation
- Automatically sets padding token if not defined
- Supports configurable sequence lengths

### Training
- Uses cosine learning rate scheduler
- Implements gradient accumulation for effective batch size control
- Supports mixed precision training (FP16)
- Includes weight decay for regularization
- Provides TensorBoard integration for monitoring training progress

## Output

The trained model and tokenizer will be saved in the directory specified in the configuration file under `output.dir`. Training logs can be viewed using TensorBoard.

## Notes

- The script defaults to FP16 training for efficiency
- The padding token is set to the EOS token if not already defined
- Dataset column names are automatically handled during tokenization
- Training progress can be monitored through TensorBoard logs
