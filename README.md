# DeepSeek Translator App

A Streamlit-based web application for translating documents using the DeepSeek API. This application provides a user-friendly interface for translating text and documents while maintaining their original formatting.

## Features

- Text translation with support for multiple languages
- Modern and responsive user interface
- Real-time translation preview
- Support for various document formats
- Customizable translation settings

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run deepseek_translator_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the interface to:
   - Enter text for translation
   - Select source and target languages
   - Upload documents for translation
   - View and download translated content

## Configuration

The application can be configured through the `streamlit.toml` file. Key settings include:

- Theme customization
- Server settings
- Browser behavior
- Runner options

## Requirements

- Python 3.8+
- Streamlit
- DeepSeek API access
- Other dependencies listed in requirements.txt

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 