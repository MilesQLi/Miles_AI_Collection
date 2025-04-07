# Miles' Open Source AI Application Collection

A curated collection of open-source applications powered by foundation models, designed to demonstrate practical applications of AI in everyday tools.

## Quick Start

You can launch any application in the collection using our convenient launcher:

- **Windows**: Double click `launcher.bat`
- **Linux/Mac**: Run `./launcher.sh`

![Application Launcher](launcher.png)

Simply select an application from the list and click "Launch Application" to start using it.

## Applications

### 1. AI Text Completion App (auto_completion_app/)

A desktop application that provides real-time AI-powered text completion suggestions as you type, supporting both OpenAI and Ollama language models. The app features a clean, intuitive interface with customizable settings for different AI models and completion behaviors. Perfect for writers, developers, or anyone looking to enhance their typing efficiency with AI assistance.

![AI Text Completion App Screenshot](./auto_completion_app/screenshot.png)

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


## Launch

Launch the application collection:
- Windows: Double click `launcher.bat`
- Linux/Mac: 
  ```bash
  chmod +x launcher.sh
  ./launcher.sh
  ```

## Configuration

The app can be configured through the Settings menu, which allows you to set:

- API Type (OpenAI or Ollama)
- API URL
- API Key (for OpenAI)
- Model name
- Completion delay (in seconds)
- Number of tokens to generate
- Auto-completion toggle

Settings are automatically saved to `~/.aicompletion.json`

### Default API URLs
- OpenAI: `http://127.0.0.1:5000/v1/completions`
- Ollama: `http://localhost:11434/api/generate`

## Usage

1. Launch the application
2. Configure your API settings through the Settings menu
3. Start typing in the text editor
4. After the configured delay, AI completion suggestions will appear in gray
5. Press Tab to accept the suggestion, or continue typing to ignore it
6. Use the File menu to create new files, open existing ones, or save your work

## Key Shortcuts

- **Tab**: Accept the current completion suggestion
- **Backspace**: Remove current completion suggestion
- **Enter**: Remove current completion and start a new line

## File Operations

- **New**: Create a new empty document
- **Open**: Open an existing text file
- **Save**: Save the current document
- **Save As**: Save the document with a new name

## Notes

- The application requires either an OpenAI-compatible API server or Ollama running locally
- For OpenAI, you need to provide a valid API key in the settings
- Auto-completion can be toggled on/off using the checkbox in the status bar
- Completion suggestions appear in gray text and can be accepted or ignored

## License

Apache 2.0

## Contributing

Contributions are welcome! If you have ideas for new AI applications or improvements to existing ones, please feel free to submit a Pull Request.