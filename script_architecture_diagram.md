# Layout Parser Script Architecture Diagram

## 📋 **Overall Flow**

```
PDF Input → Layout Detection → Text Extraction → Translation → Image Generation
    ↓              ↓               ↓              ↓              ↓
sample.pdf → Detectron2 Model → OCR (Tesseract) → DeepL API → translated_layout.png
```

## 🏗️ **Main Components Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYOUT PARSER SCRIPT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   INPUT LAYER   │    │ PROCESSING LAYER│    │  OUTPUT LAYER   │        │
│  │                 │    │                 │    │                 │        │
│  │ • PDF File      │───▶│ • Layout Model  │───▶│ • JSON Results  │        │
│  │ • Config        │    │ • OCR Engine    │    │ • Text File     │        │
│  │ • API Keys      │    │ • Translation   │    │ • Images        │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Detailed Processing Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STEP-BY-STEP EXECUTION                            │
└─────────────────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ┌─────────────────┐
   │ main()          │
   │ • Load model    │
   │ • Setup logging │
   │ • Validate PDF  │
   └─────────┬───────┘
             │
             ▼
2. PDF PROCESSING
   ┌─────────────────┐
   │process_pdf_pages│
   │ • Convert to    │
   │   images        │
   │ • Extract styles│
   │ • Detect layout │
   └─────────┬───────┘
             │
             ▼
3. LAYOUT DETECTION
   ┌─────────────────┐
   │ Detectron2      │
   │ • Find blocks   │
   │ • Classify types│
   │ • Get coords    │
   └─────────┬───────┘
             │
             ▼
4. BLOCK PROCESSING
   ┌─────────────────┐
   │merge_overlapping│
   │split_multi_sect │
   │process_parallel │
   └─────────┬───────┘
             │
             ▼
5. TEXT EXTRACTION
   ┌─────────────────┐
   │extract_text_    │
   │from_block       │
   │ • OCR with      │
   │   Tesseract     │
   │ • Clean text    │
   └─────────┬───────┘
             │
             ▼
6. TRANSLATION
   ┌─────────────────┐
   │translate_text   │
   │ • DeepL API     │
   │ • Error handling│
   │ • Caching       │
   └─────────┬───────┘
             │
             ▼
7. IMAGE GENERATION
   ┌─────────────────┐
   │create_translated│
   │_image           │
   │ • Font sizing   │
   │ • Text layout   │
   │ • Styling       │
   └─────────┬───────┘
             │
             ▼
8. OUTPUT GENERATION
   ┌─────────────────┐
   │ Save Results    │
   │ • JSON file     │
   │ • Text file     │
   │ • PNG images    │
   └─────────────────┘
```

## 🧩 **Core Function Blocks**

### **1. Configuration & Setup**
```
┌─────────────────────────────────────────┐
│ CONFIGURATION BLOCK                     │
├─────────────────────────────────────────┤
│ • logging.basicConfig()                 │
│ • CACHE_DIR setup                       │
│ • Exception classes                     │
│ • @cache_result decorator               │
└─────────────────────────────────────────┘
```

### **2. PDF Processing**
```
┌─────────────────────────────────────────┐
│ PDF PROCESSING BLOCK                    │
├─────────────────────────────────────────┤
│ • convert_pdf_to_image()                │
│   └─ pdf2image.convert_from_path()      │
│ • get_text_style()                      │
│   └─ PyMuPDF font extraction            │
│ • validate_pdf()                        │
└─────────────────────────────────────────┘
```

### **3. Layout Detection**
```
┌─────────────────────────────────────────┐
│ LAYOUT DETECTION BLOCK                  │
├─────────────────────────────────────────┤
│ • lp.Detectron2LayoutModel()            │
│   └─ PubLayNet model                    │
│ • Block types: Text, Title, List,       │
│   Table, Figure                         │
│ • Coordinate scaling                    │
└─────────────────────────────────────────┘
```

### **4. Block Processing**
```
┌─────────────────────────────────────────┐
│ BLOCK PROCESSING BLOCK                  │
├─────────────────────────────────────────┤
│ • merge_overlapping_blocks()            │
│   └─ Combine related blocks             │
│ • split_multi_section_blocks()          │
│   └─ Separate logical sections          │
│ • should_split_block()                  │
│   └─ Detect multiple sections           │
│ • create_split_blocks()                 │
│   └─ Create new block boundaries        │
└─────────────────────────────────────────┘
```

### **5. OCR & Text Extraction**
```
┌─────────────────────────────────────────┐
│ OCR & TEXT EXTRACTION BLOCK             │
├─────────────────────────────────────────┤
│ • extract_text_from_block()             │
│   ├─ Image preprocessing                │
│   │  ├─ CLAHE enhancement               │
│   │  ├─ Adaptive thresholding           │
│   │  ├─ Morphological operations        │
│   │  └─ Denoising                       │
│   ├─ Multiple OCR attempts              │
│   │  └─ Different PSM modes             │
│   └─ Text scoring & selection           │
│ • optimize_image_for_ocr()              │
│ • clean_text()                          │
└─────────────────────────────────────────┘
```

### **6. Translation**
```
┌─────────────────────────────────────────┐
│ TRANSLATION BLOCK                       │
├─────────────────────────────────────────┤
│ • translate_text()                      │
│   ├─ Text chunking (4000 char limit)    │
│   ├─ DeepL API calls                    │
│   ├─ Error handling                     │
│   └─ Result caching                     │
│ • @cache_result decorator               │
└─────────────────────────────────────────┘
```

### **7. Font & Styling**
```
┌─────────────────────────────────────────┐
│ FONT & STYLING BLOCK                    │
├─────────────────────────────────────────┤
│ • calculate_optimal_font_size_for_      │
│   filling()                             │
│   ├─ Test font sizes 14-46              │
│   ├─ Simulate text wrapping             │
│   ├─ Check height constraints           │
│   └─ Find common denominator            │
│ • get_original_style_info()             │
│   └─ Extract bold/italic flags          │
│ • get_font_set()                        │
│   └─ Cross-platform font loading        │
└─────────────────────────────────────────┘
```

### **8. Image Generation**
```
┌─────────────────────────────────────────┐
│ IMAGE GENERATION BLOCK                  │
├─────────────────────────────────────────┤
│ • create_translated_image()             │
│   ├─ Create white background            │
│   ├─ Draw colored blocks                │
│   ├─ Calculate text positioning         │
│   ├─ Apply font styling                 │
│   ├─ Handle text alignment              │
│   └─ Add block type labels              │
│ • Text wrapping algorithm               │
│ • Color schemes by block type           │
└─────────────────────────────────────────┘
```

## 🔧 **Utility Functions**

```
┌─────────────────────────────────────────┐
│ UTILITY FUNCTIONS                       │
├─────────────────────────────────────────┤
│ • safe_ocr()                            │
│ • safe_translation()                    │
│ • safe_image_processing()               │
│ • is_similar_fuzzy()                    │
│ • extract_table_data()                  │
│ • extract_list_items()                  │
│ • validate_pdf()                        │
│ • validate_output_path()                │
│ • validate_image()                      │
└─────────────────────────────────────────┘
```

## 📊 **Data Flow Diagram**

```
INPUT DATA                 PROCESSING STAGES                OUTPUT DATA
┌─────────┐               ┌─────────────────┐              ┌─────────┐
│sample.  │──────────────▶│                 │─────────────▶│layout_  │
│pdf      │               │                 │              │results. │
└─────────┘               │                 │              │json     │
                          │                 │              └─────────┘
┌─────────┐               │                 │              ┌─────────┐
│DeepL    │──────────────▶│   MAIN SCRIPT   │─────────────▶│extracted│
│API Key  │               │                 │              │_text.txt│
└─────────┘               │                 │              └─────────┘
                          │                 │              
┌─────────┐               │                 │              ┌─────────┐
│Font     │──────────────▶│                 │─────────────▶│translated│
│Files    │               │                 │              │_layout. │
└─────────┘               └─────────────────┘              │png      │
                                                           └─────────┘
                                                           ┌─────────┐
                                                           │layout_  │
                                                           │visualiz.│
                                                           │png      │
                                                           └─────────┘
```

## 🎯 **Key Algorithms**

### **Block Splitting Algorithm**
```
1. Extract text from large block
2. Find section markers (INLOGGEGEVENS, SEPA, IBAN, etc.)
3. Calculate text positions
4. Split block coordinates proportionally
5. Assign appropriate block types
6. Store pre-extracted text
```

### **Font Sizing Algorithm**
```
1. For each block and text combination:
   a. Test font sizes from 14 to 46
   b. Simulate text wrapping
   c. Check if text fits in 95% of block height
   d. Record maximum working font size
2. Take minimum across all blocks (common denominator)
3. Ensure size is within 16-40 range
```

### **OCR Enhancement Algorithm**
```
1. Apply multiple preprocessing methods:
   - CLAHE contrast enhancement
   - Adaptive thresholding
   - Morphological operations
   - Denoising
2. Try different PSM modes (3,4,6,7,8)
3. Score results based on:
   - Text length
   - Recognizable patterns
   - Special character penalty
4. Select best result
```

## 🧪 **Testing Framework**

```
┌─────────────────────────────────────────┐
│ TESTING BLOCK                           │
├─────────────────────────────────────────┤
│ • TestLayoutParser class                │
│   ├─ test_image_optimization()          │
│   ├─ test_text_similarity()             │
│   ├─ test_table_extraction()            │
│   ├─ test_list_extraction()             │
│   └─ test_pdf_validation()              │
│ • Temporary file creation               │
│ • Resource cleanup                      │
└─────────────────────────────────────────┘
```

## 📈 **Performance Optimizations**

```
┌─────────────────────────────────────────┐
│ PERFORMANCE OPTIMIZATIONS               │
├─────────────────────────────────────────┤
│ • @cache_result decorator               │
│   └─ Disk-based caching with pickle    │
│ • concurrent.futures.ThreadPoolExecutor│
│   └─ Parallel block processing         │
│ • Multiple OCR attempts                 │
│   └─ Best result selection             │
│ • Efficient text wrapping              │
│   └─ Word-by-word calculation          │
└─────────────────────────────────────────┘
```

This diagram shows how the script is organized into logical blocks, with clear data flow and processing stages. Each component has a specific responsibility and they work together to transform a PDF into a translated, properly formatted layout. 