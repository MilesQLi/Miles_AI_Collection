# Miles' Open Source AI Application Collection

A curated collection of open-source applications powered by foundation models, designed to demonstrate practical applications of AI in everyday tools.

## Quick Start

You can launch any application in the collection using our convenient launcher:

- **Windows**: Double click `launcher.bat`
- **Linux/Mac**: Run `./launcher.sh`

![Application Launcher](launcher.png)

The launcher provides an easy-to-use interface to browse and start any application in the collection.

## Applications

### 1. AI Text Completion App (auto_completion_app/)

A desktop application that provides real-time AI-powered text completion suggestions as you type, supporting both OpenAI and Ollama language models. The app features a clean, intuitive interface with customizable settings for different AI models and completion behaviors. Perfect for writers, developers, or anyone looking to enhance their typing efficiency with AI assistance.

![AI Text Completion App Screenshot](auto_completion_app/screenshot.png)

### 2. Intelligent Linux Web Console (Linux_intelligent_web_console/)

A web-based Linux terminal console powered by LangChain and LLMs that helps users execute commands through natural language understanding. The console combines traditional terminal functionality with AI assistance to make command-line operations more accessible.

![Intelligent Linux Web Console Screenshot](Linux_intelligent_web_console/screenshot.png)

### 3. Lightning Fabric Distributed Pretraining (LightningFabricDistributedPretraining/)

A flexible framework for distributed pretraining of language models using Lightning Fabric with FSDP (Fully Sharded Data Parallel) strategy. This application allows you to pretrain any Hugging Face model on any dataset with a simple YAML configuration. It supports multi-GPU and multi-node training with optimized performance for modern NVIDIA GPUs with Tensor Cores.

### 4. HuggingFace SFT Training Script (HuggingFace_SFT/)

A powerful script for conducting distributed fine-tuning of language models using Supervised Fine-Tuning (SFT) with configurable parameters loaded from a YAML file.  It properly sets up the training mechanics so as no to train on input part. It is a perfect template for researchers and developers looking to fine-tune language models on their specific datasets with minimal setup.

## Coming Soon

More AI-powered applications are in development and will be added to this collection. Stay tuned for applications in areas such as:
- Image Generation
- Code Assistance
- Document Analysis
- Audio Processing
- And more!

## Requirements

- Python 3.x
- tkinter (usually comes with Python)
- Additional requirements specific to each application

## Manual Launch

If you prefer to run applications directly:

1. Navigate to the application directory
2. Run the application using Python:
   ```bash
   python main.py
   ```

## Contributing

Contributions are welcome! If you have ideas for new AI applications or improvements to existing ones, please feel free to submit a Pull Request.

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Users are responsible for checking and validating the correctness of their configuration files, safetensor files, and binary files generated using the software. The developers assume no responsibility for any errors, omissions, or other issues coming in these files, or any consequences resulting from the use of these files.

## License

Apache 2.0