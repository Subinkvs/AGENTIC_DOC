# Agentic Document Extraction Demo

A powerful CLI tool that leverages Landing AI's Vision Agent to extract and analyze information from documents and images using natural language prompts.

## Overview

This project demonstrates how to use Landing AI's Vision Agent to:
- Extract text and data from documents
- Analyze images with custom instructions
- Display the extracted information directly in the CLI
- Save extraction results to JSON files for further processing

The Vision Agent can understand natural language prompts and extract information from documents and images according to your specific needs.

## Features

- **Natural Language Interface**: Describe what you want to extract in plain English
- **Document Analysis**: Extract text, tables, forms, and structured data from PDFs and images
- **Image Processing**: Count objects, identify features, and analyze visual content
- **CLI Output**: View extraction results directly in your terminal
- **JSON Export**: Save extraction results to JSON files for further processing

## Requirements

- Python 3.11 or higher
- Landing AI API key
- Poetry for dependency management

## Installation

### Quick Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/agentic_document_extraction_demo.git
   cd agentic_document_extraction_demo
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up your API key:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file to add your Landing AI API key.

### Alternative Setup with Conda

If you prefer to use conda:

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate agentic_document_extraction_demo
   ```

2. Install dependencies with poetry:
   ```bash
   poetry install
   ```

3. Configure your API key:
   ```bash
   cp .env.template .env
   ```
   Edit the `.env` file to add your Landing AI API key.

## Usage

### Basic Usage

```bash
python extract.py path/to/your/document.pdf
```

### Custom Extraction

Specify what you want to extract with the `--prompt` parameter:

```bash
python extract.py invoice.pdf --prompt "Extract the invoice number, date, and total amount from this invoice."
```

### Save Results to File

To save the extraction results to a JSON file:

```bash
python extract.py document.pdf --output results.json
```

### Verbose Output

For detailed logging:

```bash
python extract.py document.pdf --verbose
```

## Example Use Cases

### Document Analysis
```bash
python extract.py ./sample_documents/TSLA-q4.pdf --prompt "WHat is the total model 3 sales for q4 2023?"
```


## Project Structure

```
agentic_document_extraction_demo/
├── extract.py              # Main script for document extraction
├── environment.yml         # Conda environment specification
├── pyproject.toml          # Poetry configuration
├── .env.template           # Template for environment variables
├── .env                    # Environment variables (with API keys)
└── .gitignore              # Git ignore file
```

## How It Works

1. The script takes a document/image file and a prompt as input
2. It sends the file and prompt to Landing AI's Vision Agent
3. The Vision Agent processes the document according to the prompt
4. Results are displayed in the CLI and optionally saved to a JSON file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses the Landing AI Vision Agent library
- Special thanks to the Landing AI team for their powerful vision AI tools
