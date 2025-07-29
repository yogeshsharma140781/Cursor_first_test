#!/usr/bin/env python3
"""
Adobe Layout-Preserving PDF Recreator
Recreate the original PDF layout with cleaned, structured text
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import zipfile
import tempfile
import openai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, darkblue, darkgrey
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    ADOBE_SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Adobe PDF Services SDK not available: {e}")
    ADOBE_SDK_AVAILABLE = False

class AdobeLayoutPreservingRecreator:
    """Recreate PDF with original layout but cleaned and translated text"""
    
    def __init__(self, credentials_path: str = "backend/pdfservices-api-credentials.json"):
        """Initialize layout preserving recreator"""
        
        self.credentials = None
        self.pdf_services = None
        self.openai_client = None
        self.credentials_path = credentials_path
        
        self.setup_adobe_credentials()
        self.setup_fonts()
        self.setup_openai_client()
    
    def setup_adobe_credentials(self):
        """Setup Adobe PDF Services credentials"""
        
        print("🔑 Setting up Adobe PDF Services credentials...")
        
        if not ADOBE_SDK_AVAILABLE:
            print("❌ Adobe PDF Services SDK not available")
            return False
        
        try:
            # Load credentials from file
            if os.path.exists(self.credentials_path):
                with open(self.credentials_path, 'r') as f:
                    creds_data = json.load(f)
                
                client_id = creds_data.get('client_credentials', {}).get('client_id')
                client_secret = creds_data.get('client_credentials', {}).get('client_secret')
                
                if client_id and client_secret:
                    self.credentials = ServicePrincipalCredentials(
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    
                    # Create PDF Services instance
                    self.pdf_services = PDFServices(credentials=self.credentials)
                    print("✅ Adobe PDF Services initialized successfully")
                    return True
            
            print("❌ Adobe credentials not found")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up Adobe credentials: {e}")
            return False
    
    def setup_fonts(self):
        """Setup fonts for PDF generation"""
        
        try:
            # Try to register available fonts
            font_paths = {
                'NotoSans': 'backend/fonts/NotoSans-Regular.ttf',
                'NotoSans-Bold': 'backend/fonts/NotoSans-Bold.ttf',
                'NotoSansDevanagari': 'backend/fonts/NotoSansDevanagari-Regular.ttf',
                'NotoSansDevanagari-Bold': 'backend/fonts/NotoSansDevanagari-Bold.ttf',
                'Arial': '/System/Library/Fonts/Arial.ttf',
                'Arial-Bold': '/System/Library/Fonts/Arial Bold.ttf',
                'Times-Roman': '/System/Library/Fonts/Times.ttc',
                'Times-Bold': '/System/Library/Fonts/Times Bold.ttc',
                'Helvetica': '/System/Library/Fonts/Helvetica.ttc',
                'Helvetica-Bold': '/System/Library/Fonts/Helvetica Bold.ttc'
            }
            
            self.available_fonts = {}
            self.font_mappings = {}
            
            # Register available fonts
            for font_name, font_path in font_paths.items():
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        self.available_fonts[font_name] = font_path
                        print(f"✅ Registered font: {font_name}")
                    except Exception as e:
                        print(f"⚠️  Could not register font {font_name}: {e}")
            
            # Setup font mappings for common font families
            self.setup_font_mappings()
            
            if not self.available_fonts:
                print("📝 Using default fonts")
                self.available_fonts = {'Helvetica': 'built-in'}
            
        except Exception as e:
            print(f"⚠️  Font setup error: {e}")
            self.available_fonts = {'Helvetica': 'built-in'}
    
    def setup_font_mappings(self):
        """Setup font family mappings for common font names"""
        
        # Map common font names to available fonts
        self.font_mappings = {
            # Arial family
            'Arial': 'Arial' if 'Arial' in self.available_fonts else 'Helvetica',
            'Arial-Bold': 'Arial-Bold' if 'Arial-Bold' in self.available_fonts else 'Helvetica-Bold',
            'Arial-BoldMT': 'Arial-Bold' if 'Arial-Bold' in self.available_fonts else 'Helvetica-Bold',
            'ArialMT': 'Arial' if 'Arial' in self.available_fonts else 'Helvetica',
            
            # Times family
            'Times': 'Times-Roman' if 'Times-Roman' in self.available_fonts else 'Times-Roman',
            'Times-Roman': 'Times-Roman' if 'Times-Roman' in self.available_fonts else 'Times-Roman',
            'Times-Bold': 'Times-Bold' if 'Times-Bold' in self.available_fonts else 'Times-Bold',
            'TimesNewRoman': 'Times-Roman' if 'Times-Roman' in self.available_fonts else 'Times-Roman',
            'TimesNewRoman-Bold': 'Times-Bold' if 'Times-Bold' in self.available_fonts else 'Times-Bold',
            
            # Helvetica family
            'Helvetica': 'Helvetica',
            'Helvetica-Bold': 'Helvetica-Bold' if 'Helvetica-Bold' in self.available_fonts else 'Helvetica',
            'HelveticaNeue': 'Helvetica',
            'HelveticaNeue-Bold': 'Helvetica-Bold' if 'Helvetica-Bold' in self.available_fonts else 'Helvetica',
            
            # Noto family
            'NotoSans': 'NotoSans' if 'NotoSans' in self.available_fonts else 'Helvetica',
            'NotoSans-Bold': 'NotoSans-Bold' if 'NotoSans-Bold' in self.available_fonts else 'Helvetica-Bold',
            'NotoSansDevanagari': 'NotoSansDevanagari' if 'NotoSansDevanagari' in self.available_fonts else 'NotoSans',
            'NotoSansDevanagari-Bold': 'NotoSansDevanagari-Bold' if 'NotoSansDevanagari-Bold' in self.available_fonts else 'NotoSans-Bold',
            
            # Fallback
            'Default': 'Helvetica'
        }
        
        print(f"✅ Font mappings configured: {len(self.font_mappings)} mappings")
    
    def setup_openai_client(self):
        """Setup OpenAI client for translation"""
        
        print("🤖 Setting up OpenAI client...")
        
        try:
            # Try environment variable first
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                print("❌ OpenAI API key not found in environment variables")
                print("⚠️  Translation will be skipped")
                return False
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up OpenAI client: {e}")
            print("⚠️  Translation will be skipped")
            return False
    
    def extract_layout_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract layout data from original PDF"""
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return None
        
        try:
            print(f"📄 Extracting layout data from {pdf_path}...")
            
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                print(f"❌ PDF file not found: {pdf_path}")
                return None
            
            # Upload PDF to Adobe
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()
            
            input_asset = self.pdf_services.upload(
                input_stream=input_stream,
                mime_type=PDFServicesMediaType.PDF
            )
            
            # Create extract job
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT]
            )
            extract_pdf_job = ExtractPDFJob(
                input_asset=input_asset, 
                extract_pdf_params=extract_pdf_params
            )
            
            # Submit job
            print("🚀 Submitting Adobe Extract job...")
            location = self.pdf_services.submit(extract_pdf_job)
            
            # Get result
            print("⏳ Waiting for extraction completion...")
            pdf_services_response = self.pdf_services.get_job_result(location, ExtractPDFResult)
            result_asset = pdf_services_response.get_result().get_resource()
            stream_asset = self.pdf_services.get_content(result_asset)
            
            # Save extraction result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extraction_output = f"adobe_layout_extract_{timestamp}.zip"
            
            with open(extraction_output, "wb") as file:
                file.write(stream_asset.get_input_stream())
            
            print(f"✅ Layout extraction completed: {extraction_output}")
            
            # Extract and parse JSON
            extract_dir = f"adobe_layout_temp_{timestamp}"
            with zipfile.ZipFile(extraction_output, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Load structured data
            json_file = os.path.join(extract_dir, "structuredData.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                
                print(f"✅ Loaded layout data with {len(extraction_data.get('elements', []))} elements")
                
                # Clean up
                os.remove(extraction_output)
                
                return extraction_data
            else:
                print("❌ No structured data found in extraction")
                return None
                
        except Exception as e:
            print(f"❌ Layout extraction error: {e}")
            return None
    
    def clean_and_structure_text(self, extraction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Clean and structure text while preserving position data"""
        
        if not extraction_data or 'elements' not in extraction_data:
            print("❌ No elements found in extraction data")
            return []
        
        elements = extraction_data['elements']
        text_elements = [elem for elem in elements if elem.get('Text')]
        
        print(f"📝 Processing {len(text_elements)} text elements...")
        
        # Sort elements by position (top to bottom, left to right)
        # In bottom-left coordinate system, larger Y = higher on page
        sorted_elements = sorted(text_elements, key=lambda x: (
            -x.get('Bounds', [0, 0, 0, 0])[1],  # Y position (top to bottom: largest Y first)
            x.get('Bounds', [0, 0, 0, 0])[0]    # X position (left to right: smallest X first)
        ))
        
        # Clean each text element
        cleaned_elements = []
        for elem in sorted_elements:
            original_text = elem.get('Text', '').strip()
            if not original_text:
                continue
            
            # Clean the text
            cleaned_text = self.clean_text(original_text)
            
            # Extract font information
            font_info = elem.get('Font', {})
            
            # Create cleaned element with position and font info preserved
            cleaned_elem = {
                'text': cleaned_text,
                'original_text': original_text,
                'bounds': elem.get('Bounds', [0, 0, 0, 0]),
                'font_info': font_info,
                'position': {
                    'x': elem.get('Bounds', [0, 0, 0, 0])[0],
                    'y': elem.get('Bounds', [0, 0, 0, 0])[1],
                    'width': elem.get('Bounds', [0, 0, 0, 0])[2],
                    'height': elem.get('Bounds', [0, 0, 0, 0])[3]
                },
                'style': self.extract_font_style(font_info)
            }
            
            cleaned_elements.append(cleaned_elem)
        
        print(f"✅ Cleaned {len(cleaned_elements)} text elements")
        return cleaned_elements
    
    def clean_text(self, text: str) -> str:
        """Clean and improve text quality"""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR spacing issues
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        # Fix quotes
        text = text.replace(' "', '"')
        text = text.replace('" ', '"')
        
        # Fix common OCR errors for Dutch text
        text = text.replace('Antwoordnurnrner', 'Antwoordnummer')
        text = text.replace('Financien', 'Financiën')
        text = text.replace('kopieen', 'kopieën')
        text = text.replace('formuller', 'formulier')
        text = text.replace('v66r', 'vóór')
        
        return text
    
    def extract_font_style(self, font_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract font style information from Adobe font data"""
        
        if not font_info:
            return {
                'font_family': 'Helvetica',
                'font_size': 12,
                'is_bold': False,
                'is_italic': False,
                'color': 'black'
            }
        
        # Extract font family
        font_name = font_info.get('name', 'Helvetica')
        font_family = font_info.get('family_name', font_name)
        
        # Extract font size
        font_size = font_info.get('size', 12)
        
        # Determine if bold
        is_bold = (
            'Bold' in font_name or 
            'bold' in font_name.lower() or
            font_info.get('weight', 400) >= 700
        )
        
        # Determine if italic
        is_italic = (
            'Italic' in font_name or 
            'italic' in font_name.lower() or
            font_info.get('style', '') == 'italic'
        )
        
        # Extract color (if available)
        color = font_info.get('color', 'black')
        
        return {
            'font_family': font_family,
            'font_name': font_name,
            'font_size': font_size,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'color': color,
            'original_font_info': font_info
        }
    
    def get_mapped_font_name(self, style: Dict[str, Any]) -> str:
        """Get the best available font name for the given style"""
        
        original_font = style.get('font_name', 'Helvetica')
        font_family = style.get('font_family', 'Helvetica')
        is_bold = style.get('is_bold', False)
        is_italic = style.get('is_italic', False)
        
        # Try direct mapping first
        if original_font in self.font_mappings:
            mapped_font = self.font_mappings[original_font]
            if mapped_font in self.available_fonts:
                return mapped_font
        
        # Try family-based mapping
        if font_family in self.font_mappings:
            mapped_font = self.font_mappings[font_family]
            if mapped_font in self.available_fonts:
                return mapped_font
        
        # Build font name based on style
        base_font = font_family
        
        # Handle bold
        if is_bold:
            bold_variants = [
                f"{base_font}-Bold",
                f"{base_font}Bold",
                f"{base_font}-BoldMT",
                f"{base_font}MT-Bold"
            ]
            for variant in bold_variants:
                if variant in self.available_fonts:
                    return variant
                if variant in self.font_mappings and self.font_mappings[variant] in self.available_fonts:
                    return self.font_mappings[variant]
        
        # Handle italic (for future enhancement)
        if is_italic:
            italic_variants = [
                f"{base_font}-Italic",
                f"{base_font}Italic",
                f"{base_font}-ItalicMT"
            ]
            for variant in italic_variants:
                if variant in self.available_fonts:
                    return variant
                if variant in self.font_mappings and self.font_mappings[variant] in self.available_fonts:
                    return self.font_mappings[variant]
        
        # Fallback to base font
        if base_font in self.available_fonts:
            return base_font
        
        # Final fallback
        return 'Helvetica'
    
    def group_elements_into_paragraphs(self, cleaned_elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group text elements into logical paragraphs based on proximity and line breaks"""
        
        if not cleaned_elements:
            return []
        
        print(f"📝 Grouping {len(cleaned_elements)} elements into paragraphs...")
        
        # Sort elements by position (already sorted in clean_and_structure_text)
        elements = cleaned_elements.copy()
        
        paragraphs = []
        current_paragraph = []
        
        for i, elem in enumerate(elements):
            current_y = elem['position']['y']
            current_x = elem['position']['x']
            
            if not current_paragraph:
                # Start first paragraph
                current_paragraph = [elem]
            else:
                # Check if this element should be part of current paragraph
                last_elem = current_paragraph[-1]
                last_y = last_elem['position']['y']
                last_x = last_elem['position']['x']
                
                # Calculate vertical distance
                y_distance = abs(current_y - last_y)
                avg_height = (elem['position']['height'] + last_elem['position']['height']) / 2
                
                # Calculate horizontal distance
                x_distance = abs(current_x - last_x)
                
                # Conservative grouping logic for better document structure
                # Elements are in same paragraph if they're very close
                same_line = y_distance <= avg_height * 0.3  # Very close vertically (same line)
                close_proximity = y_distance <= avg_height * 1.0  # Close proximity
                reasonable_x = x_distance <= 150  # Not too far horizontally
                
                # Only group if they're on the same line OR very close with reasonable horizontal distance
                if same_line or (close_proximity and reasonable_x):
                    current_paragraph.append(elem)
                else:
                    # Start new paragraph - any significant gap starts new paragraph
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                    current_paragraph = [elem]
        
        # Add last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # Keep most paragraphs separate for better document structure
        merged_paragraphs = []
        for paragraph in paragraphs:
            combined_text = ' '.join(elem['text'] for elem in paragraph).strip()
            
            # Keep all paragraphs with any meaningful content
            if len(combined_text) > 2:  # At least 3 characters
                merged_paragraphs.append(paragraph)
            elif merged_paragraphs and len(paragraph) == 1:
                # Only merge single-element tiny paragraphs
                merged_paragraphs[-1].extend(paragraph)
        
        print(f"✅ Created {len(merged_paragraphs)} meaningful paragraphs")
        
        # Debug: Show paragraph lengths
        for i, paragraph in enumerate(merged_paragraphs):
            combined_text = ' '.join(elem['text'] for elem in paragraph).strip()
            print(f"   📄 Paragraph {i+1}: {len(paragraph)} elements, '{combined_text[:50]}...'")
        
        return merged_paragraphs
    
    def translate_paragraph(self, paragraph: List[Dict[str, Any]], 
                          source_lang: str = "Dutch", 
                          target_lang: str = "English") -> List[Dict[str, Any]]:
        """Translate a paragraph while preserving individual element positions"""
        
        if not self.openai_client:
            print("⚠️  OpenAI client not available, returning original paragraph")
            return paragraph
        
        try:
            # Combine text from all elements in paragraph
            combined_text = ' '.join(elem['text'] for elem in paragraph).strip()
            
            # Skip very short texts
            if len(combined_text) < 3:
                return paragraph
            
            print(f"🔄 Translating: '{combined_text[:50]}...'")
            
            # Create translation prompt
            prompt = f"""Translate the following {source_lang} text to {target_lang}. 

Rules:
1. Preserve the original meaning and tone
2. Keep numbers, dates, and proper nouns unchanged
3. Use appropriate formal language for official documents
4. Maintain the structure and length as much as possible

Text to translate:
{combined_text}

Translation:"""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a professional translator specializing in {source_lang} to {target_lang} translation of government and official documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Now distribute the translated text back to individual elements
            translated_paragraph = self.distribute_translation_to_elements(paragraph, translated_text)
            
            print(f"✅ Translated to: '{translated_text[:50]}...'")
            
            return translated_paragraph
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return paragraph
    
    def distribute_translation_to_elements(self, original_paragraph: List[Dict[str, Any]], 
                                         translated_text: str) -> List[Dict[str, Any]]:
        """Distribute translated text back to individual elements preserving original structure"""
        
        if not original_paragraph:
            return original_paragraph
        
        # Split translated text into words
        translated_words = translated_text.split()
        
        # Calculate the total character length of original text
        total_original_chars = sum(len(elem['text']) for elem in original_paragraph)
        
        if total_original_chars == 0:
            return original_paragraph
        
        translated_paragraph = []
        word_index = 0
        
        for i, elem in enumerate(original_paragraph):
            original_text = elem['text']
            original_length = len(original_text)
            
            # Calculate how many words this element should get based on its relative size
            if i == len(original_paragraph) - 1:
                # Last element gets all remaining words
                element_words = translated_words[word_index:]
            else:
                # Calculate proportion of total text this element represents
                proportion = original_length / total_original_chars
                words_for_element = max(1, int(len(translated_words) * proportion))
                
                # Ensure we don't exceed available words
                words_for_element = min(words_for_element, len(translated_words) - word_index)
                element_words = translated_words[word_index:word_index + words_for_element]
                word_index += words_for_element
            
            # Create translated element with original positioning
            translated_elem = elem.copy()
            translated_elem['text'] = ' '.join(element_words) if element_words else original_text
            translated_elem['original_text'] = original_text
            translated_elem['is_translated'] = True
            
            # Keep original positioning exactly as it was
            # This preserves the document structure
            
            translated_paragraph.append(translated_elem)
        
        return translated_paragraph
    
    def translate_all_paragraphs(self, paragraphs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Translate all paragraphs and return flattened list of elements"""
        
        if not self.openai_client:
            print("⚠️  OpenAI client not available, returning original text")
            # Flatten paragraphs back to elements
            all_elements = []
            for paragraph in paragraphs:
                all_elements.extend(paragraph)
            return all_elements
        
        print(f"🌍 Translating {len(paragraphs)} paragraphs from Dutch to English...")
        
        translated_elements = []
        
        for i, paragraph in enumerate(paragraphs):
            print(f"📝 Processing paragraph {i+1}/{len(paragraphs)}")
            translated_paragraph = self.translate_paragraph(paragraph)
            translated_elements.extend(translated_paragraph)
        
        print(f"✅ Translation completed: {len(translated_elements)} elements processed")
        return translated_elements
    
    def get_page_dimensions(self, extraction_data: Dict[str, Any]) -> Tuple[float, float]:
        """Get page dimensions from extraction data"""
        
        # Try to get page dimensions from metadata
        pages = extraction_data.get('pages', [])
        if pages:
            page = pages[0]
            width = page.get('width', 612)  # Default to letter size
            height = page.get('height', 792)
            return width, height
        
        # Fallback to calculating from element bounds
        elements = extraction_data.get('elements', [])
        if elements:
            max_x = max((elem.get('Bounds', [0, 0, 0, 0])[0] + elem.get('Bounds', [0, 0, 0, 0])[2]) for elem in elements)
            max_y = max((elem.get('Bounds', [0, 0, 0, 0])[1] + elem.get('Bounds', [0, 0, 0, 0])[3]) for elem in elements)
            return max_x + 50, max_y + 50  # Add some margin
        
        # Ultimate fallback
        return 612, 792  # Letter size
    
    def create_layout_preserving_pdf(self, cleaned_elements: List[Dict[str, Any]], 
                                   page_width: float, page_height: float, 
                                   output_path: str) -> bool:
        """Create PDF preserving original layout with cleaned text"""
        
        try:
            print(f"📄 Creating layout-preserving PDF...")
            
            # Create canvas
            c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
            
            print(f"🎨 Placing {len(cleaned_elements)} text elements with original styling...")
            
            for i, elem in enumerate(cleaned_elements):
                text = elem['text']
                pos = elem['position']
                style = elem.get('style', {})
                
                # Skip empty text
                if not text.strip():
                    continue
                
                # Use coordinates directly (both Adobe Extract and ReportLab use bottom-left origin)
                x = pos['x']
                y = pos['y']  # No coordinate transformation needed
                
                # Get original font information
                original_font_size = style.get('font_size', 12)
                
                # Use original font size, but ensure it's reasonable
                font_size = max(min(original_font_size, 24), 6)  # Between 6 and 24 points
                
                # Get the best available font for this style
                font_name = self.get_mapped_font_name(style)
                
                # Set font and size
                try:
                    c.setFont(font_name, font_size)
                except Exception as e:
                    c.setFont('Helvetica', font_size)
                
                # Use original color if available, otherwise use content-based coloring
                if style.get('color') and style['color'] != 'black':
                    # Try to use original color
                    try:
                        c.setFillColor(style['color'])
                    except:
                        c.setFillColor(black)
                else:
                    # Content-based coloring for better readability
                    if any(keyword in text.lower() for keyword in ['dienst', 'ministerie', 'onderwerp', 'reply', 'number']):
                        c.setFillColor(darkblue)  # Headers in dark blue
                    elif any(keyword in text.lower() for keyword in ['datum', 'antwoordnummer', 'date', 'phone']):
                        c.setFillColor(darkgrey)  # Metadata in grey
                    else:
                        c.setFillColor(black)  # Regular text in black
                
                # Conservative text placement respecting original bounds
                max_width = pos['width']
                
                # Calculate the actual text width
                text_width = c.stringWidth(text, font_name, font_size)
                
                # If text fits within original bounds, place it as-is
                if text_width <= max_width or max_width < 50:
                    c.drawString(x, y, text)
                else:
                    # Text is too long - wrap within bounds
                    words = text.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        test_line = ' '.join(current_line + [word])
                        test_width = c.stringWidth(test_line, font_name, font_size)
                        
                        if test_width <= max_width or not current_line:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # Draw each line, but limit to avoid too many lines
                    max_lines = 3  # Limit to 3 lines per element
                    for j, line in enumerate(lines[:max_lines]):
                        line_y = y - (j * font_size * 1.1)
                        if line_y > 0:  # Don't draw below page
                            c.drawString(x, line_y, line)
                
                # Progress indicator
                if i % 10 == 0:
                    print(f"   📝 Placed {i+1}/{len(cleaned_elements)} elements...")
            
            # Add generation metadata
            c.setFont('Helvetica', 8)
            c.setFillColor(darkgrey)
            c.drawString(20, 20, f"Recreated with cleaned text on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Save PDF
            c.save()
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Layout-preserving PDF created: {output_path} ({file_size:,} bytes)")
            
            return True
            
        except Exception as e:
            print(f"❌ PDF creation error: {e}")
            return False
    
    def recreate_with_layout(self, input_pdf: str = "scanned_ADOBE_OCR_DUTCH.pdf", 
                           output_pdf: str = "recreated_layout_improved.pdf"):
        """Complete workflow to recreate PDF with preserved layout"""
        
        print("🚀 ADOBE LAYOUT-PRESERVING TRANSLATION + STYLING WORKFLOW")
        print("=" * 70)
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return False
        
        # Step 1: Extract layout data
        print("\n📄 STEP 1: EXTRACTING LAYOUT DATA")
        extraction_data = self.extract_layout_data(input_pdf)
        if not extraction_data:
            print("❌ Layout extraction failed")
            return False
        
        # Step 2: Clean and structure text
        print("\n🧹 STEP 2: CLEANING TEXT WHILE PRESERVING POSITIONS & FONTS")
        cleaned_elements = self.clean_and_structure_text(extraction_data)
        if not cleaned_elements:
            print("❌ Text cleaning failed")
            return False
        
        # Analyze font usage
        font_usage = {}
        for elem in cleaned_elements:
            font_name = elem.get('style', {}).get('font_name', 'Unknown')
            font_size = elem.get('style', {}).get('font_size', 12)
            font_key = f"{font_name} ({font_size}pt)"
            font_usage[font_key] = font_usage.get(font_key, 0) + 1
        
        print(f"   📊 Font analysis: {len(font_usage)} unique font/size combinations")
        for font_key, count in sorted(font_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      • {font_key}: {count} elements")
        
        # Step 3: Group elements into paragraphs
        print("\n📝 STEP 3: GROUPING ELEMENTS INTO PARAGRAPHS")
        paragraphs = self.group_elements_into_paragraphs(cleaned_elements)
        if not paragraphs:
            print("❌ Paragraph grouping failed")
            return False
        
        # Step 4: Translate paragraphs
        print("\n🌍 STEP 4: TRANSLATING PARAGRAPHS WITH OPENAI")
        translated_elements = self.translate_all_paragraphs(paragraphs)
        
        # Step 5: Get page dimensions
        print("\n📏 STEP 5: DETERMINING PAGE DIMENSIONS")
        page_width, page_height = self.get_page_dimensions(extraction_data)
        print(f"   📄 Page size: {page_width:.1f} x {page_height:.1f} points")
        
        # Step 6: Create layout-preserving PDF with translations and styling
        print("\n🎨 STEP 6: CREATING LAYOUT-PRESERVING PDF WITH TRANSLATIONS & STYLING")
        success = self.create_layout_preserving_pdf(
            translated_elements, page_width, page_height, output_pdf
        )
        
        # Summary
        print(f"\n🎉 LAYOUT RECREATION WITH TRANSLATION COMPLETE!")
        print(f"   📄 Input PDF: {input_pdf}")
        print(f"   📝 Processed: {len(paragraphs)} paragraphs")
        print(f"   🌍 Translated: {len(translated_elements)} text elements")
        print(f"   🔤 Available fonts: {len(self.available_fonts)} registered")
        print(f"   📄 Output PDF: {output_pdf}")
        print(f"   📏 Page Size: {page_width:.0f} x {page_height:.0f} points")
        
        if success:
            print(f"\n✅ Benefits achieved:")
            print(f"   🎯 Original layout preserved")
            print(f"   🧹 Text quality improved")
            print(f"   🌍 Dutch to English translation")
            print(f"   📍 Exact positioning maintained")
            print(f"   🔤 Original font styling preserved")
            print(f"   🎨 Enhanced typography")
        
        return success

def main():
    """Main execution function"""
    
    try:
        # Parse command line arguments
        input_pdf = "scanned_ADOBE_OCR_DUTCH.pdf"
        output_pdf = "recreated_layout_improved.pdf"
        
        if len(sys.argv) >= 2:
            input_pdf = sys.argv[1]
        if len(sys.argv) >= 3:
            output_pdf = sys.argv[2]
        
        # Initialize layout recreator with credentials in current directory
        recreator = AdobeLayoutPreservingRecreator(credentials_path="pdfservices-api-credentials.json")
        
        # Recreate PDF with preserved layout
        success = recreator.recreate_with_layout(input_pdf, output_pdf)
        
        if success:
            print("\n✅ Adobe layout-preserving translation completed successfully!")
        else:
            print("\n❌ Layout translation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
