# Quick Start Guide - PDF Translation to English

## 🚀 Get Started in 3 Steps

### Step 1: Set up your API keys
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-openai-api-key-here'

# Ensure Adobe credentials file exists
ls pdfservices-api-credentials.json
```

### Step 2: Install dependencies
```bash
pip install -r requirements_adobe_openai.txt
```

### Step 3: Run the translation
```bash
# Test the setup first
python test_translation_workflow.py

# Translate a single PDF
python run_translation_workflow.py your_document.pdf

# Or translate all PDFs in the directory
python batch_translate_pdfs.py
```

## 📄 What You'll Get

- **Input**: `your_document.pdf` (any language)
- **Output**: `your_document_translated_to_english.pdf` (English)
- **Features**: 
  - ✅ All text translated to English
  - ✅ Original layout preserved
  - ✅ Images and logos kept in place
  - ✅ Fonts and formatting maintained

## 🎯 Example Output

```
🚀 Starting PDF Translation Workflow
📄 Input: sample2.pdf
📄 Output: sample2_translated_to_english.pdf
🌐 Target Language: English

⏳ Processing... This may take a few minutes...
=== Starting Complete PDF Translation Workflow ===
Extracting elements from: sample2.pdf
⏳ Uploading PDF to Adobe PDF Services...
⏳ Submitting extract job...
⏳ Waiting for job to complete...
Extraction complete. Found 45 elements
Parsed 45 text elements (sorted by visual reading order)
Translating 45 text blocks to English
Reconstructing PDF with translated text and original formatting...
PDF reconstructed with original formatting (black text only): sample2_translated_to_english.pdf
=== Workflow Complete ===

✅ Translation complete!
📄 Output saved to: sample2_translated_to_english.pdf

🎉 Your PDF has been successfully translated to English!
```

## 🔧 Troubleshooting

### Common Issues:

1. **"OPENAI_API_KEY not set"**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. **"Adobe credentials file not found"**
   - Download credentials from [Adobe PDF Services](https://www.adobe.com/go/dcsdks_credentials)
   - Save as `pdfservices-api-credentials.json`

3. **Missing dependencies**
   ```bash
   pip install pdfservices-sdk openai reportlab requests PyMuPDF
   ```

## 📚 More Information

- **Full Documentation**: See `README_Translation_Workflow.md`
- **Test Script**: Run `python test_translation_workflow.py` to verify setup
- **Batch Processing**: Use `python batch_translate_pdfs.py` for multiple files

## 💡 Tips

- Start with a simple PDF to test
- Check console output for any warnings
- Large PDFs may take 2-5 minutes to process
- Ensure you have sufficient OpenAI API credits 