# Layout Parser Script - Function Map & Code Blocks

## 📍 **Function Location Map** (Line Numbers)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCRIPT STRUCTURE (1845 lines)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ IMPORTS & SETUP (1-40)                                                     │
│ ├─ Standard libraries (logging, sys, typing, etc.)                         │
│ ├─ Third-party libraries (layoutparser, cv2, PIL, etc.)                    │
│ └─ Configuration (logging, cache setup)                                    │
│                                                                             │
│ EXCEPTION CLASSES (43-56)                                                  │
│ ├─ LayoutParserError (base)                                                │
│ ├─ OCRError                                                                │
│ ├─ TranslationError                                                        │
│ └─ ImageProcessingError                                                    │
│                                                                             │
│ CORE FUNCTIONS (59-1620)                                                   │
│ ├─ cache_result() (59-85)                                                  │
│ ├─ translate_text() (86-138)                                               │
│ ├─ convert_pdf_to_image() (139-144)                                        │
│ ├─ get_text_style() (145-222)                                              │
│ ├─ process_sections() (223-325)                                            │
│ ├─ clean_text() (326-388)                                                  │
│ ├─ optimize_image_for_ocr() (389-433)                                      │
│ ├─ process_pdf_pages() (434-517)                                           │
│ ├─ extract_text_from_block() (518-656)                                     │
│ ├─ merge_overlapping_blocks() (657-753)                                    │
│ ├─ get_font_set() (754-826)                                                │
│ ├─ create_translated_image() (827-1264)                                    │
│ ├─ safe_* functions (1265-1296)                                            │
│ ├─ process_blocks_parallel() (1297-1480)                                   │
│ ├─ create_split_blocks() (1481-1602)                                       │
│ ├─ should_split_block() (1603-1623)                                        │
│ └─ split_multi_section_blocks() (1624-1645)                                │
│                                                                             │
│ TESTING FRAMEWORK (1646-1720)                                              │
│ └─ TestLayoutParser class with 5 test methods                              │
│                                                                             │
│ VALIDATION FUNCTIONS (1721-1759)                                           │
│ ├─ validate_pdf()                                                          │
│ ├─ validate_output_path()                                                  │
│ └─ validate_image()                                                        │
│                                                                             │
│ MAIN EXECUTION (1760-1845)                                                 │
│ └─ main() function with error handling                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔗 **Function Call Hierarchy**

```
main() [1760]
│
├─ lp.Detectron2LayoutModel() [Load AI model]
│
├─ validate_pdf() [1721]
│
├─ validate_output_path() [1734] (for each output file)
│
└─ process_pdf_pages() [434]
   │
   ├─ convert_from_path() [pdf2image library]
   │
   ├─ get_text_style() [145]
   │  └─ fitz.open() [PyMuPDF]
   │
   ├─ model.detect() [Detectron2 AI model]
   │
   ├─ merge_overlapping_blocks() [657]
   │
   ├─ split_multi_section_blocks() [1624]
   │  ├─ extract_text_from_block() [518]
   │  ├─ should_split_block() [1603]
   │  └─ create_split_blocks() [1481]
   │
   ├─ process_blocks_parallel() [1297]
   │  └─ ThreadPoolExecutor
   │     └─ process_block() [1365]
   │        ├─ extract_text_from_block() [518]
   │        │  ├─ optimize_image_for_ocr() [389]
   │        │  └─ pytesseract.image_to_string()
   │        ├─ detect_block_type() [1302]
   │        ├─ clean_text() [326]
   │        └─ translate_text() [86]
   │           └─ DeepL API call
   │
   ├─ lp.draw_box() [Create visualization]
   │
   └─ create_translated_image() [827]
      ├─ calculate_optimal_font_size_for_filling() [851]
      ├─ get_original_style_info() [943]
      ├─ get_font_set() [754]
      └─ PIL.ImageDraw operations
```

## 🎨 **Visual Function Blocks**

### **🔧 Setup & Configuration Block**
```
┌─────────────────────────────────────────┐
│ Lines 1-85: SETUP & CONFIGURATION      │
├─────────────────────────────────────────┤
│                                         │
│ cache_result() [59-85]                  │
│ ┌─────────────────────────────────────┐ │
│ │ @functools.wraps(func)              │ │
│ │ def wrapper(*args, **kwargs):       │ │
│ │   • Create cache key from args      │ │
│ │   • Try to load from .cache/        │ │
│ │   • Execute function if not cached  │ │
│ │   • Save result to cache            │ │
│ │   • Return result                   │ │
│ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **🌐 Translation Block**
```
┌─────────────────────────────────────────┐
│ Lines 86-138: TRANSLATION               │
├─────────────────────────────────────────┤
│                                         │
│ @cache_result                           │
│ translate_text(text, target_lang="EN")  │
│ ┌─────────────────────────────────────┐ │
│ │ • Clean & normalize text            │ │
│ │ • Split into 4000-char chunks       │ │
│ │ • For each chunk:                   │ │
│ │   ├─ POST to DeepL API              │ │
│ │   ├─ Handle errors gracefully       │ │
│ │   └─ Collect translated chunks      │ │
│ │ • Combine & restore formatting      │ │
│ │ • Return result                   │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **📄 PDF Processing Block**
```
┌─────────────────────────────────────────┐
│ Lines 139-433: PDF PROCESSING           │
├─────────────────────────────────────────┤
│                                         │
│ convert_pdf_to_image() [139-144]        │
│ ┌─────────────────────────────────────┐ │
│ │ pdf2image.convert_from_path()       │ │
│ │ return np.array(first_page)         │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ get_text_style() [145-222]              │
│ ┌─────────────────────────────────────┐ │
│ │ • Open PDF with PyMuPDF             │ │
│ │ • Extract font info per block       │ │
│ │ • Detect text alignment             │ │
│ │ • Analyze paragraph structure       │ │
│ │ • Return style dictionary           │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ optimize_image_for_ocr() [389-433]      │
│ ┌─────────────────────────────────────┐ │
│ │ • Convert to grayscale              │ │
│ │ • Apply CLAHE contrast enhancement  │ │
│ │ • Denoise with fastNlMeansDenoising │ │
│ │ • Adaptive thresholding             │ │
│ │ • Deskew if needed                  │ │
│ │ • Morphological noise removal       │ │
│ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **🤖 AI Layout Detection Block**
```
┌─────────────────────────────────────────┐
│ Lines 434-517: LAYOUT DETECTION         │
├─────────────────────────────────────────┤
│                                         │
│ process_pdf_pages() [434-517]           │
│ ┌─────────────────────────────────────┐ │
│ │ For each page:                      │ │
│ │ ├─ Convert PDF page to image        │ │
│ │ ├─ Extract text styles              │ │
│ │ ├─ model.detect(image) [AI model]   │ │
│ │ ├─ Scale coordinates to image size  │ │
│ │ ├─ Merge overlapping blocks         │ │
│ │ ├─ Split multi-section blocks       │ │
│ │ ├─ Process blocks in parallel       │ │
│ │ ├─ Create visualization             │ │
│ │ └─ Create translated image          │ │
│ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **👁️ OCR Text Extraction Block**
```
┌─────────────────────────────────────────┐
│ Lines 518-656: OCR TEXT EXTRACTION      │
├─────────────────────────────────────────┤
│                                         │
│ extract_text_from_block() [518-656]     │
│ ┌─────────────────────────────────────┐ │
│ │ • Crop image to block coordinates   │ │
│ │ • Apply 4 preprocessing methods:    │ │
│ │   ├─ CLAHE + OTSU threshold         │ │
│ │   ├─ Adaptive threshold             │ │
│ │   ├─ Morphological operations       │ │
│ │   └─ Denoise + OTSU threshold       │ │
│ │ • Try 5 PSM modes per method        │ │
│ │ • Score results by:                 │ │
│ │   ├─ Text length                    │ │
│ │   ├─ Recognizable patterns          │ │
│ │   └─ Special character penalty      │ │
│ │ • Select best result               │ │
│ │ • Apply OCR error corrections       │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **🔗 Block Processing Block**
```
┌─────────────────────────────────────────┐
│ Lines 657-1645: BLOCK PROCESSING        │
├─────────────────────────────────────────┤
│                                         │
│ merge_overlapping_blocks() [657-753]    │
│ ┌─────────────────────────────────────┐ │
│ │ • Sort blocks by position           │ │
│ │ • Check overlap conditions:         │ │
│ │   ├─ Direct coordinate overlap      │ │
│ │   ├─ Vertical proximity             │ │
│ │   ├─ Horizontal alignment           │ │
│ │   └─ Similar heights                │ │
│ │ • Merge qualifying blocks           │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
│ split_multi_section_blocks() [1624]     │
│ ┌─────────────────────────────────────┐ │
│ │ For each block:                     │ │
│ │ ├─ Extract text                     │ │
│ │ ├─ should_split_block()             │ │
│ │ └─ create_split_blocks() if needed  │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
│ create_split_blocks() [1481-1602]       │
│ ┌─────────────────────────────────────┐ │
│ │ • Find section markers:             │ │
│ │   ├─ INLOGGEGEVENS                  │ │
│ │   ├─ SEPA MACHTIGING                │ │
│ │   ├─ IBAN: / NL83                   │ │
│ │   └─ BEVESTIGING                    │ │
│ │ • Calculate text positions          │ │
│ │ • Split block coordinates           │ │
│ │ • Assign block types                │ │
│ │ • Store pre-extracted text          │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **🎨 Font & Image Generation Block**
```
┌─────────────────────────────────────────┐
│ Lines 754-1264: FONT & IMAGE GENERATION │
├─────────────────────────────────────────┤
│                                         │
│ calculate_optimal_font_size() [851-942] │
│ ┌─────────────────────────────────────┐ │
│ │ For each block:                     │ │
│ │ ├─ Test font sizes 14-46            │ │
│ │ ├─ Simulate text wrapping           │ │
│ │ ├─ Check 95% height constraint      │ │
│ │ └─ Record max working size          │ │
│ │ Take minimum across all blocks      │ │
│ │ Ensure size in range 16-40          │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
│ create_translated_image() [827-1264]    │
│ ┌─────────────────────────────────────┐ │
│ │ • Create white background           │ │
│ │ • For each block:                   │ │
│ │   ├─ Draw colored background        │ │
│ │   ├─ Calculate optimal font size    │ │
│ │   ├─ Load appropriate font style    │ │
│ │   ├─ Wrap text to fit width         │ │
│ │   ├─ Position text with alignment   │ │
│ │   ├─ Draw text with proper color    │ │
│ │   └─ Add block type label           │ │
│ │ • Return final PIL Image            │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### **⚡ Parallel Processing Block**
```
┌─────────────────────────────────────────┐
│ Lines 1297-1480: PARALLEL PROCESSING    │
├─────────────────────────────────────────┤
│                                         │
│ process_blocks_parallel() [1297-1480]   │
│ ┌─────────────────────────────────────┐ │
│ │ • Sort blocks by position           │ │
│ │ • Group into rows                   │ │
│ │ • ThreadPoolExecutor:               │ │
│ │   └─ process_block() for each       │ │
│ │      ├─ Extract/use pre-text        │ │
│ │      ├─ Detect block type           │ │
│ │      ├─ Clean text                  │ │
│ │      └─ Translate text              │ │
│ │ • Collect results                   │ │
│ │ • Sort by reading order             │ │
│ │ └─────────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

## 🔄 **Data Transformation Flow**

```
INPUT                    TRANSFORMATIONS                     OUTPUT
┌─────────┐             ┌─────────────────┐                ┌─────────┐
│ PDF     │────────────▶│ pdf2image       │───────────────▶│ numpy   │
│ File    │             │ convert_from_   │                │ array   │
│         │             │ path()          │                │ (image) │
└─────────┘             └─────────────────┘                └─────────┘
                                 │
                                 ▼
┌─────────┐             ┌─────────────────┐                ┌─────────┐
│ Image   │────────────▶│ Detectron2      │───────────────▶│ Layout  │
│ Array   │             │ model.detect()  │                │ Blocks  │
└─────────┘             └─────────────────┘                └─────────┘
                                 │
                                 ▼
┌─────────┐             ┌─────────────────┐                ┌─────────┐
│ Layout  │────────────▶│ extract_text_   │───────────────▶│ Raw     │
│ Blocks  │             │ from_block()    │                │ Text    │
└─────────┘             └─────────────────┘                └─────────┘
                                 │
                                 ▼
┌─────────┐             ┌─────────────────┐                ┌─────────┐
│ Raw     │────────────▶│ translate_text()│───────────────▶│ Trans-  │
│ Text    │             │ (DeepL API)     │                │ lated   │
└─────────┘             └─────────────────┘                └─────────┘
                                 │
                                 ▼
┌─────────┐             ┌─────────────────┐                ┌─────────┐
│ Trans-  │────────────▶│ create_         │───────────────▶│ Final   │
│ lated   │             │ translated_     │                │ PNG     │
│ Text    │             │ image()         │                │ Image   │
└─────────┘             └─────────────────┘                └─────────┘
```

This comprehensive diagram shows exactly how the 1845-line script is organized, with specific line numbers, function relationships, and data flow. Each block has a clear responsibility and they work together to transform a PDF into a beautifully formatted, translated layout. 