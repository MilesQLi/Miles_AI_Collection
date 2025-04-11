# AI Text Completion App

A desktop application that provides real-time AI-powered text completion suggestions as you type, supporting both OpenAI and Ollama language models.

![AI Text Completion App Screenshot](screenshot.png)

## Features

- Real-time AI text completion suggestions
- Support for both OpenAI and Ollama APIs
- Customizable completion delay and token length
- Easy completion acceptance with Tab key
- File operations (New, Open, Save, Save As)
- Configurable settings
- Dark text completion preview
- Status bar with completion state indicators

## Requirements

- Python 3.x
- tkinter (usually comes with Python)
- requests library (`pip install requests`)

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
   ```bash
   pip install requests
   ```
3. Run the application:
   ```bash
   python main.py
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

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Users are responsible for checking and validating the correctness of their configuration files, safetensor files, and binary files generated using the software. The developers assume no responsibility for any errors, omissions, or other issues coming in these files, or any consequences resulting from the use of these files.


## License

Apache 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.