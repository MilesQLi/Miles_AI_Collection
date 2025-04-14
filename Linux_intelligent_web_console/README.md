# Intelligent Linux Web Console

A web-based Linux terminal console powered by LangChain and LLMs that helps users execute commands through natural language understanding.

## Features

- 🌐 Web-based terminal interface
- 🤖 AI-powered command suggestions using LangChain
- 💻 Real-time command execution
- 📁 Directory navigation support
- 🔒 Secure command execution with confirmation prompts

![Screenshot](screenshot.png)

## How It Works

The console allows you to:
1. Execute standard Linux commands directly
2. Get AI assistance for command creation using natural language

### AI Command Assistant

To get help creating a command, use the `ask` keyword followed by your request in quotes:

```bash
ask "Find all PDF files in the current directory and its subdirectories"
```

The AI will:
1. Process your request
2. Provide an explanation
3. Suggest the exact command
4. Ask for confirmation before execution

## Technical Stack

- **Backend**: FastAPI
- **AI Integration**: LangChain with Ollama
- **WebSocket**: Real-time communication
- **Model**: Qwen 2.5 (14B-instruct)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install fastapi langchain-openai uvicorn
```

3. Run the server:
```bash
python main.py
```

4. Access the web console at: `http://localhost:5001`

## Environment Setup

The application requires:
- Python 3.7+
- Ollama running locally with Qwen 2.5 model
- Access to a Linux environment

## Usage Examples

1. Directory listing with AI:
```bash
ask "Show me all files and their sizes in a human-readable format"
```

2. Finding specific files:
```bash
ask "Find all Python files modified in the last 24 hours"
```

3. System information:
```bash
ask "Show me system memory usage in a human-readable format"
```

## Security Considerations

- All AI-suggested commands require user confirmation before execution
- Commands are executed in a controlled environment
- Working directory is isolated to `/workspace/` by default

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Users are responsible for checking and validating the correctness of their configuration files, safetensor files, and binary files generated using the software. The developers assume no responsibility for any errors, omissions, or other issues coming in these files, or any consequences resulting from the use of these files.


## License

Apache 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
