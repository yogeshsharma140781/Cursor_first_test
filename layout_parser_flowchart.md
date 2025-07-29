# Layout Parser Script Flowchart

```mermaid
flowchart TD
    A[Start: main()] --> B{Check Arguments}
    B -->|--test| C[Run Unit Tests]
    B -->|No args| D[Initialize Logging]
    
    C --> C1[TestLayoutParser.setUpClass]
    C1 --> C2[Run test_image_optimization]
    C2 --> C3[Run test_text_similarity]
    C3 --> C4[Run test_table_extraction]
    C4 --> C5[Run test_list_extraction]
    C5 --> C6[Run test_pdf_validation]
    C6 --> C7[TestLayoutParser.tearDownClass]
    C7 --> END[End]
    
    D --> E[Load Detectron2 Layout Model]
    E --> F[Set PDF Path: 'sample.pdf']
    F --> G[validate_pdf()]
    G -->|Invalid| ERROR1[Error: Invalid PDF]
    G -->|Valid| H[validate_output_path()]
    H --> I[process_pdf_pages()]
    
    I --> J[Convert PDF to Images<br/>convert_from_path()]
    J --> K[Open PDF with PyMuPDF]
    K --> L{For Each Page}
    
    L --> M[Convert PIL Image to NumPy Array]
    M --> N[Convert to RGB if needed]
    N --> O[get_text_style()]
    
    O --> O1[Extract text blocks from PDF]
    O1 --> O2[Analyze font, size, flags, color]
    O2 --> O3[Detect text alignment]
    O3 --> O4[Determine paragraph structure]
    O4 --> O5[Return text_styles dict]
    
    O5 --> P[Layout Detection<br/>model.detect()]
    P --> Q[Scale Coordinates to Image Size]
    Q --> R[merge_overlapping_blocks()]
    
    R --> R1{Check Block Overlap}
    R1 -->|Significant Overlap| R2[Merge Blocks]
    R1 -->|No Overlap| R3[Keep Separate]
    R2 --> R4[Create Merged Block]
    R3 --> R4
    R4 --> R5{More Blocks?}
    R5 -->|Yes| R1
    R5 -->|No| S[process_blocks_parallel()]
    
    S --> T{For Each Block}
    T --> U[extract_text_and_spans_from_block()]
    
    U --> U1{Text Styles Available?}
    U1 -->|Yes| U2[Find Best Overlapping Style Block]
    U1 -->|No| U3[Use OCR Fallback]
    U2 --> U4[Extract Formatted Text with Spans]
    U3 --> U5[optimize_image_for_ocr()]
    U4 --> U6[join_split_words()]
    U5 --> U6
    U6 --> V[Process Block Type]
    
    V --> V1{Block Type?}
    V1 -->|Table| V2[extract_table_data()]
    V1 -->|List| V3[extract_list_items()]
    V1 -->|Text/Title/Figure| V4[Continue with Text]
    
    V2 --> W[Translation Process]
    V3 --> W
    V4 --> W
    
    W --> W1{Span Lines Available?}
    W1 -->|Yes| W2[translate_spans_with_formatting()]
    W1 -->|No| W3[safe_translation()]
    
    W2 --> W4[Preserve Bold/Italic Formatting]
    W3 --> W5[postprocess_translation()]
    W4 --> W6[match_casing()]
    W5 --> W6
    W6 --> X[Create Result Object]
    
    X --> Y{More Blocks?}
    Y -->|Yes| T
    Y -->|No| Z[Check for Missed Text Style Blocks]
    
    Z --> Z1{Uncovered Style Blocks?}
    Z1 -->|Yes| Z2[Create Synthetic Blocks]
    Z1 -->|No| AA[Remove Duplicates]
    Z2 --> Z3[Process Synthetic Blocks]
    Z3 --> AA
    
    AA --> AA1[is_similar_fuzzy() comparison]
    AA1 --> AA2[Check coordinate overlap]
    AA2 --> AA3[Keep best version of duplicates]
    AA3 --> BB[Create Visualizations]
    
    BB --> BB1[lp.draw_box() - Layout Visualization]
    BB1 --> BB2[create_translated_image()]
    
    BB2 --> BB3[Clear original text areas]
    BB3 --> BB4[get_font_set() - Load appropriate fonts]
    BB4 --> BB5[Draw translated text with formatting]
    BB5 --> CC[Save Page Results]
    
    CC --> DD{More Pages?}
    DD -->|Yes| L
    DD -->|No| EE[Save Final Results]
    
    EE --> FF[Save JSON: layout_results.json]
    FF --> GG[Save Text: extracted_text.txt]
    GG --> HH[Success Message]
    HH --> END
    
    ERROR1 --> ERROR2[Log Error]
    ERROR2 --> ERROR3[Print User-Friendly Message]
    ERROR3 --> ERROR4[Exit with Code 1]
    ERROR4 --> END
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef cache fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class A,END startEnd
    class D,E,F,I,J,K,M,N,O,P,Q,R,S,U,V,W,X,Z,AA,BB,CC,EE,FF,GG,HH process
    class B,L,T,Y,DD,U1,V1,W1,Z1,R1,R5 decision
    class ERROR1,ERROR2,ERROR3,ERROR4 error
    class O1,O2,O3,O4,O5,R2,R3,R4,U2,U3,U4,U5,U6,V2,V3,V4,W2,W3,W4,W5,W6,Z2,Z3,AA1,AA2,AA3,BB1,BB2,BB3,BB4,BB5 process
```

## Key Components Explained:

### 1. **Main Entry Points**
- **main()**: Primary execution path for PDF processing
- **Unit Tests**: Alternative execution path when `--test` argument is provided

### 2. **PDF Processing Pipeline**
- **Validation**: PDF file and output path validation
- **Conversion**: PDF to images using `pdf2image`
- **Style Extraction**: Font, formatting, and layout information using PyMuPDF
- **Layout Detection**: Using Detectron2 model to identify text blocks, tables, lists, etc.

### 3. **Text Processing**
- **Block Merging**: Combines overlapping or related text blocks
- **Text Extraction**: Extracts text with formatting information (bold, italic)
- **OCR Fallback**: Uses Tesseract when PDF text extraction fails
- **Word Joining**: Fixes split words across lines

### 4. **Translation Pipeline**
- **Span-based Translation**: Preserves formatting during translation
- **DeepL API**: External translation service with caching
- **Post-processing**: Fixes common translation errors
- **Case Matching**: Maintains original text casing patterns

### 5. **Output Generation**
- **Deduplication**: Removes similar or overlapping results
- **Visualization**: Creates annotated images showing detected layout
- **Translated Images**: Generates new images with translated text
- **Multiple Formats**: JSON and text file outputs

### 6. **Error Handling & Validation**
- **Custom Exceptions**: Specific error types for different failure modes
- **Safe Functions**: Wrapper functions with error handling
- **Validation**: Input validation for PDFs, images, and paths
- **Logging**: Comprehensive logging throughout the process

### 7. **Caching System**
- **@cache_result**: Decorator for caching expensive operations
- **Disk-based**: Persistent caching using pickle files
- **Hash-based Keys**: Cache keys based on function arguments 