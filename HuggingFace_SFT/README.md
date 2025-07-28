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
- PEFT (for LoRA support)

Install the requirements:
```bash
pip install -r requirements.txt
```

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
   - `use_lora`: Whether to enable LoRA training (default: false)
   - `lora_config`: LoRA configuration parameters:
     - `r`: LoRA rank (default: 16)
     - `lora_alpha`: LoRA alpha parameter (default: 32)
     - `target_modules`: List of modules to apply LoRA to (default: ["q_proj", "v_proj"])
     - `lora_dropout`: LoRA dropout rate (default: 0.1)
     - `bias`: Bias handling ("none", "all", or "lora_only")
     - `task_type`: Task type for PEFT (default: "CAUSAL_LM")

2. **Dataset Configuration**:
   - `name`: Dataset name from HuggingFace
   - `subset`: Dataset subset (optional)
   - `split`: Dataset split to use

3. **Training Configuration**:
   - Various training parameters like batch size, learning rate, etc.

## Usage

### Full Fine-tuning (Default)
Run the script with:

```bash
accelerate launch sft.py --config default_train_config.yaml
```

### LoRA Fine-tuning
To use LoRA for efficient fine-tuning:

```bash
accelerate launch sft.py --config lora_train_config.yaml
```

If no config file is specified, it will use `default_train_config.yaml` by default.

## LoRA Training

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- Reduces memory usage significantly
- Speeds up training
- Allows for efficient model sharing
- Maintains good performance with fewer trainable parameters

### LoRA Configuration Tips

- **Rank (r)**: Higher values (16-64) give better performance but use more memory
- **Alpha (lora_alpha)**: Usually set to 2*r for optimal performance
- **Target Modules**: Common choices include:
  - `["q_proj", "v_proj"]` (minimal, fast)
  - `["q_proj", "v_proj", "k_proj", "o_proj"]` (balanced)
  - `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (comprehensive)

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

### LoRA Models
When using LoRA, the model will save:
- **LoRA adapter weights** in `output_dir/lora_weights/` (small, efficient for sharing)
- **Complete merged model** in `output_dir/` (full model with LoRA weights merged)
- **Tokenizer files** in `output_dir/`

This gives you both the efficient LoRA adapter for sharing and the complete merged model for direct use.

### Full Fine-tuning Models
When not using LoRA, the entire model weights will be saved (much larger file size).

## Loading Saved Models

### Loading LoRA Models
You have two options when loading LoRA-trained models:

1. **Load the merged model directly** (recommended for inference):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/output_dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/output_dir")
```

2. **Load base model + LoRA adapter** (for further training or sharing):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("path/to/base_model")
model = PeftModel.from_pretrained(base_model, "path/to/output_dir/lora_weights")
tokenizer = AutoTokenizer.from_pretrained("path/to/base_model")
```

### Loading Full Fine-tuned Models
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/output_dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/output_dir")
``` 