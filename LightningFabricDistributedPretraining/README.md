# Lightning Fabric Distributed Pretraining

This repository contains code for distributed pretraining of language models using Lightning Fabric with FSDP (Fully Sharded Data Parallel) strategy.

## Overview

The code is designed to be flexible and can be used to pretrain any Hugging Face model on any dataset. It uses Lightning Fabric for distributed training and supports various configurations through a YAML file.

## Requirements

- PyTorch
- Lightning Fabric
- Transformers
- Datasets
- PyYAML
- TensorBoard (optional, for logging)

## Installation

```bash
pip install torch lightning transformers datasets pyyaml tensorboard
```

## Usage

### Configuration

All training parameters are specified in a YAML configuration file. A default configuration file (`config.yaml`) is provided with the following structure:

```yaml
# Model Configuration
model:
  name_or_path: gpt2
  pad_token_id: null  # Will be set to eos_token_id if None

# Dataset Configuration
dataset:
  name: nthngdy/oscar-mini
  subset: unshuffled_deduplicated_en
  split: train
  max_samples: 2000
  seq_len: 128
  trust_remote_code: true

# Training Configuration
training:
  epochs: 1
  seed: 42
  batch_size_per_device: 4
  grad_accum_steps: 4
  learning_rate: 5.0e-5
  warmup_steps: 100
  log_interval: 10

# Output Configuration
output:
  dir: ./fabric_fsdp_gpt2_output

# Lightning Fabric Configuration
fabric:
  accelerator: auto
  strategy: fsdp
  devices: auto
  precision: bf16-mixed
  num_nodes: 1
```

You can modify this file to change any of the parameters for your specific use case.

> **Note:** When editing the YAML file, make sure numeric values are not enclosed in quotes. The code includes a function to convert string values to numbers, but it's best practice to format the YAML correctly.

### Running Training

To start training, simply run:

```bash
python train.py --config path/to/your/config.yaml
```

## Performance Optimization

The code automatically sets the appropriate precision for Tensor Cores on NVIDIA GPUs:

```python
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
```

This optimization improves performance on modern NVIDIA GPUs with Tensor Cores.

## Distributed Training

The code supports distributed training using Lightning Fabric with FSDP strategy. The configuration file allows you to specify:

- Number of nodes (`fabric.num_nodes`)
- Devices to use (`fabric.devices`)
- Precision (`fabric.precision`)
- Strategy (`fabric.strategy`)

## Customization

### Using a Different Model

To use a different model, modify the `model.name_or_path` in the configuration file:

```yaml
model:
  name_or_path: gpt2-medium  # or any other model from Hugging Face
```

### Using a Different Dataset

To use a different dataset, modify the dataset section in the configuration file:

```yaml
dataset:
  name: your_dataset_name
  subset: your_subset_name
  split: train
  max_samples: 2000
  seq_len: 128
  trust_remote_code: true
```

## Output

The trained model and tokenizer will be saved to the directory specified in `output.dir`. The saved files include:

- The model checkpoint
- The tokenizer
- The configuration used for training

## Logging

Training metrics are logged using TensorBoard. You can view the logs by running:

```bash
tensorboard --logdir=path/to/output/dir/logs
``` 