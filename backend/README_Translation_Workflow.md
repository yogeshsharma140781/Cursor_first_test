# PDF Translation Workflow

This workflow translates PDF documents to English while preserving the original layout, formatting, and images.

## Features

- ✅ **Complete PDF Translation**: Extracts text and translates it to English
- ✅ **Layout Preservation**: Maintains original document structure and formatting
- ✅ **Image Preservation**: Keeps all images, logos, and visual elements in their original positions
- ✅ **Font Mapping**: Maps original fonts to appropriate ReportLab fonts
- ✅ **Color Preservation**: Maintains original text colors (when possible)
- ✅ **Batch Processing**: Can process multiple PDFs at once

## Prerequisites

1. **Adobe PDF Services SDK Credentials**
   - You need a `pdfservices-api-credentials.json` file
   - Get credentials from [Adobe PDF Services](https://www.adobe.com/go/dcsdks_credentials)

2. **OpenAI API Key**
   - Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **Python Dependencies**
   ```bash
   pip install -r requirements_adobe_openai.txt
   ```

## Usage

### Single PDF Translation

```bash
# Translate a specific PDF
python run_translation_workflow.py your_document.pdf

# Translate the default sample2.pdf
python run_translation_workflow.py
```

### Batch PDF Translation

```bash
# Translate all PDFs in the current directory
python batch_translate_pdfs.py

# Translate specific PDFs
python batch_translate_pdfs.py document1.pdf document2.pdf document3.pdf
```

### Direct API Usage

```python
from adobe_openai_complete_workflow import PDFTranslationWorkflow

# Initialize workflow
workflow = PDFTranslationWorkflow(
    adobe_credentials_path="pdfservices-api-credentials.json",
    openai_api_key="your-openai-api-key"
)

# Run translation
result_path = workflow.run_complete_workflow(
    input_pdf_path="input.pdf",
    output_pdf_path="output_translated.pdf",
    target_language="English"
)
```

## How It Works

1. **PDF Extraction**: Uses Adobe PDF Services SDK to extract text, images, and layout information
2. **Text Processing**: Parses and sorts text elements by visual reading order
3. **Image Extraction**: Extracts all images using both Adobe SDK and PyMuPDF
4. **Translation**: Uses OpenAI GPT-3.5-turbo to translate text to English
5. **PDF Reconstruction**: Rebuilds the PDF with translated text and original images

## Output

- **Translated PDF**: `{original_name}_translated_to_english.pdf`
- **Preserved Layout**: Original formatting, fonts, and image positions
- **English Text**: All text translated to English while maintaining structure

## Supported Languages

The workflow can translate from any language to English. The source language is automatically detected by OpenAI.

## File Structure

```
backend/
├── adobe_openai_complete_workflow.py  # Main workflow class
├── run_translation_workflow.py        # Single PDF translation script
├── batch_translate_pdfs.py           # Batch translation script
├── pdfservices-api-credentials.json  # Adobe credentials
├── sample2.pdf                       # Sample input PDF
└── fonts/                           # Custom fonts directory
    ├── NotoSans-Regular.ttf
    ├── NotoSans-Bold.ttf
    └── ...
```

## Troubleshooting

### Common Issues

1. **Missing Credentials**
   - Ensure `pdfservices-api-credentials.json` exists
   - Verify `OPENAI_API_KEY` environment variable is set

2. **Font Issues**
   - Custom fonts are automatically registered if available
   - Falls back to standard fonts if custom fonts are missing

3. **Image Placement**
   - Images without valid bounding boxes are skipped
   - Check console output for placement warnings

4. **Translation Errors**
   - Network issues with OpenAI API
   - Check API key validity and quota

### Error Messages

- `❌ Error: OPENAI_API_KEY environment variable not set`
  - Set your OpenAI API key: `export OPENAI_API_KEY='your-key'`

- `❌ Error: Adobe credentials file not found`
  - Ensure `pdfservices-api-credentials.json` is in the current directory

- `WARNING: No valid bbox for image`
  - Image will be skipped (this is normal for some PDFs)

## Examples

### Basic Translation
```bash
cd backend
python run_translation_workflow.py sample2.pdf
```

### Batch Translation
```bash
cd backend
python batch_translate_pdfs.py *.pdf
```

### Custom Output
```python
workflow = PDFTranslationWorkflow(creds_path, api_key)
workflow.run_complete_workflow(
    "input.pdf", 
    "custom_output_name.pdf", 
    "English"
)
```

## Performance

- **Processing Time**: 1-5 minutes per PDF (depending on size and complexity)
- **API Costs**: Uses OpenAI GPT-3.5-turbo for translation
- **Memory Usage**: Moderate (loads PDF into memory for processing)

## Limitations

- Text must be extractable (not pure image-based PDFs)
- Very large PDFs may require more memory
- Complex layouts with overlapping elements may need manual adjustment
- Some special characters or fonts may not render perfectly

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all prerequisites are met
3. Test with a simple PDF first
4. Check Adobe and OpenAI API quotas 