# Unicode Font Support Setup

## Problem Description

When translating PDFs to non-Roman scripts (like Hindi, Japanese, Arabic, etc.), the translated text appears as boxes (□) instead of the actual characters. This happens because the PDF generation system doesn't have the necessary Unicode fonts to render these scripts.

## Solution

This document provides a comprehensive solution to add Unicode font support to your PDF translation system.

## Quick Setup

### Automatic Setup (Recommended)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Run the setup script:
   ```bash
   ./setup_unicode_support.sh
   ```

3. Restart your translation API server

### Manual Setup

If the automatic setup doesn't work for your system, follow these steps:

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Unicode fonts:**
   ```bash
   python3 install_fonts.py
   ```

3. **Test the installation:**
   ```bash
   python3 test_unicode_fonts.py
   ```

## Supported Languages and Scripts

After setup, your PDF translation system will support:

- **Latin scripts**: English, Spanish, French, German, Italian, Portuguese, etc.
- **Chinese, Japanese, Korean (CJK)**: 中文, 日本語, 한국어
- **Devanagari scripts**: Hindi (हिंदी), Sanskrit, Nepali, Marathi
- **Arabic script**: Arabic (العربية), Persian, Urdu
- **Cyrillic script**: Russian (Русский), Ukrainian, Bulgarian
- **And many more...**

## How It Works

The solution includes several components:

### 1. Font Registration System
- Automatically detects and registers Unicode fonts on your system
- Downloads necessary fonts if not available locally
- Supports Windows, macOS, and Linux

### 2. Script Detection
- Automatically detects the writing system/script of translated text
- Uses Unicode code point ranges to identify:
  - CJK characters (Chinese, Japanese, Korean)
  - Devanagari characters (Hindi, etc.)
  - Arabic characters
  - Cyrillic characters
  - And more...

### 3. Smart Font Selection
- Selects the best available font for each language and script
- Falls back gracefully if preferred fonts are not available
- Maintains font styling (bold, italic) when possible

### 4. Enhanced PDF Generation
- Updated PDF creation to use Unicode-capable fonts
- Proper handling of right-to-left scripts (Arabic, Hebrew)
- Maintains original document layout and formatting

## Files Added/Modified

### New Files:
- `install_fonts.py` - Font download and installation script
- `test_unicode_fonts.py` - Unicode font testing script
- `setup_unicode_support.sh` - Automated setup script
- `UNICODE_FONT_SETUP.md` - This documentation

### Modified Files:
- `translator_api.py` - Enhanced with Unicode font support
- `requirements.txt` - Added requests dependency

## Testing

After setup, test your Unicode font support:

1. **Run the font test:**
   ```bash
   python3 test_unicode_fonts.py
   ```

2. **Check the generated test PDF:**
   Open `unicode_font_test.pdf` to verify that text in different languages renders correctly.

3. **Test with actual translation:**
   Try translating a PDF to Hindi, Japanese, or Arabic to see the characters render properly instead of showing boxes.

## Troubleshooting

### Problem: Fonts not found
**Solution:** Run the font installer manually:
```bash
python3 install_fonts.py
```

### Problem: Permission denied during font installation
**Solution:** 
- On Linux: Make sure you have sudo access for system font installation
- On macOS: Install fonts to user directory instead of system directory
- On Windows: Run as administrator

### Problem: Characters still showing as boxes
**Possible causes:**
1. Font registration failed - check server logs
2. Font files corrupted - re-run font installer
3. Incorrect script detection - verify with test script

### Problem: Import errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Advanced Configuration

### Custom Font Paths
You can add custom font directories by modifying the `get_system_fonts()` function in `translator_api.py`.

### Adding New Scripts
To add support for additional scripts:

1. Add the script to `LANGUAGE_SCRIPT_MAP` in `translator_api.py`
2. Add Unicode code point ranges to `detect_script_from_text()`
3. Add font configuration to `UNICODE_FONTS`

## Performance Notes

- Font registration happens once at startup
- Script detection is fast (O(n) where n is text length)
- Font selection is cached for performance
- Unicode fonts may be larger than basic fonts but provide comprehensive character support

## Security Considerations

- Font files are downloaded from official Google Fonts repository
- All downloads are verified before installation
- Fonts are installed to user directories by default (not system-wide)

## Support

If you encounter issues:

1. Check the server logs for font registration messages
2. Run the test script to verify font availability
3. Ensure you have proper write permissions for font directories
4. Verify internet connection for font downloads

## Technical Details

### Font Priority Order:
1. Language-specific fonts (e.g., Noto Sans CJK for Chinese)
2. Script-specific fonts (e.g., Noto Sans Devanagari for Hindi)
3. Universal Unicode fonts (e.g., Noto Sans)
4. System fallback fonts
5. Built-in PDF fonts (last resort)

### Supported Font Formats:
- TrueType (.ttf)
- OpenType (.otf)  
- TrueType Collection (.ttc)

This comprehensive solution ensures that your PDF translation system can handle text in virtually any language and script, eliminating the "box characters" problem. 