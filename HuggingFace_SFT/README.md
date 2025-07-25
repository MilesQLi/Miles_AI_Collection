# HuggingFace SFT Training Script

This script allows you to conduct distributed fine-tuning of language models using Supervised Fine-Tuning (SFT) with configurable parameters loaded from a YAML file. It properly sets up the training mechanics so as no to train on input part.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- TRL
- Datasets
- PyYAML
- Accelerate

## Setup

Before running the training script, you should configure the Accelerate environment. This sets up the distributed training arguments and hardware configuration. Run the following command and follow the prompts:

```bash
accelerate config
```

This only needs to be done once per environment or whenever you want to change your distributed training setup.

## Configuration

The script uses a YAML configuration file to set all parameters. A default configuration file (`default_train_config.yaml`) is provided with the current values.

### Configuration Structure

The YAML file has three main sections:

1. **Model Configuration**:
   - `path`: Path to the pre-trained model
   - `max_seq_length`: Maximum sequence length
   - `dtype`: Data type for model (null for default)
   - `load_in_4bit`: Whether to load the model in 4-bit quantization

2. **Dataset Configuration**:
   - `name`: Dataset name from HuggingFace
   - `subset`: Dataset subset (optional)
   - `split`: Dataset split to use

3. **Training Configuration**:
   - Various training parameters like batch size, learning rate, etc.

## Usage

Run the script with:

```bash
accelerate launch sft.py --config default_train_config.yaml
```



If no config file is specified, it will use `default_train_config.yaml` by default.

## Customizing the Configuration

To use a different model or dataset, create a new YAML file based on the default one and modify the parameters as needed. For example:

```yaml
# Model Configuration
model:
  path: 'your/model/path'
  max_seq_length: 256
  dtype: null
  load_in_4bit: false

# Dataset Configuration
dataset:
  name: "your/dataset"
  subset: "your_subset"
  split: "train"

# Training Configuration
training:
  gradient_accumulation_steps: 8
  per_device_train_batch_size: 2
  # ... other parameters
```

## Output

The trained model will be saved to the directory specified in the `output_dir` parameter of the training configuration. 