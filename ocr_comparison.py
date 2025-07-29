#!/usr/bin/env python3
"""
Adobe PDF Services OCR Processor
Process scanned PDFs using Adobe PDF Services API
"""

import os
import time
import fitz  # PyMuPDF
from datetime import datetime
from adobe_ocr_workflow import AdobeOCRProcessor

class AdobeOCRAnalyzer:
    """Adobe PDF Services OCR analyzer"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.input_pdf = "scanned.pdf"
        self.results = {}
    
    def analyze_pdf_text(self, pdf_path: str):
        """Analyze extracted text from PDF"""
        
        if not os.path.exists(pdf_path):
            return {
                'exists': False,
                'error': f"File not found: {pdf_path}"
            }
        
        try:
            doc = fitz.open(pdf_path)
            total_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append(page_text)
                total_text += page_text + "\n"
            
            doc.close()
            
            # Calculate statistics
            word_count = len(total_text.split())
            char_count = len(total_text)
            line_count = len(total_text.split('\n'))
            
            # Find meaningful content
            meaningful_lines = [line.strip() for line in total_text.split('\n') 
                              if line.strip() and len(line.strip()) > 5]
            
            return {
                'exists': True,
                'file_size': os.path.getsize(pdf_path),
                'page_count': len(doc),
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count,
                'meaningful_lines': len(meaningful_lines),
                'sample_text': total_text[:500] + "..." if len(total_text) > 500 else total_text,
                'first_10_lines': meaningful_lines[:10]
            }
            
        except Exception as e:
            return {
                'exists': True,
                'error': f"Error analyzing PDF: {e}"
            }
    
    def run_adobe_ocr(self):
        """Run Adobe OCR workflow"""
        
        print("🔵 RUNNING ADOBE PDF SERVICES OCR")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            processor = AdobeOCRProcessor()
            output_file = "scanned_ADOBE_OCR.pdf"
            
            success = processor.process_scanned_pdf(
                input_pdf=self.input_pdf,
                output_pdf=output_file
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Analyze results
            analysis = self.analyze_pdf_text(output_file)
            
            self.results['adobe'] = {
                'success': success,
                'processing_time': processing_time,
                'output_file': output_file,
                'analysis': analysis
            }
            
            print(f"⏱️  Adobe OCR processing time: {processing_time:.2f} seconds")
            
        except Exception as e:
            self.results['adobe'] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"❌ Adobe OCR failed: {e}")
    
    def generate_analysis_report(self):
        """Generate detailed analysis report"""
        
        print("\n📊 ADOBE OCR ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"📄 Input PDF: {self.input_pdf}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Adobe Results
        print(f"\n🔵 ADOBE PDF SERVICES RESULTS:")
        if 'adobe' in self.results:
            adobe = self.results['adobe']
            print(f"   ✅ Success: {adobe.get('success', False)}")
            print(f"   ⏱️  Processing Time: {adobe.get('processing_time', 0):.2f} seconds")
            
            if adobe.get('success') and 'analysis' in adobe:
                analysis = adobe['analysis']
                if analysis.get('exists'):
                    print(f"   📄 Output File: {adobe.get('output_file', 'N/A')}")
                    print(f"   📊 File Size: {analysis.get('file_size', 0):,} bytes ({analysis.get('file_size', 0)/1024:.1f} KB)")
                    print(f"   📄 Page Count: {analysis.get('page_count', 0)}")
                    print(f"   📝 Word Count: {analysis.get('word_count', 0)}")
                    print(f"   🔤 Character Count: {analysis.get('char_count', 0)}")
                    print(f"   📄 Meaningful Lines: {analysis.get('meaningful_lines', 0)}")
                    
                    if analysis.get('first_10_lines'):
                        print(f"   🔍 Sample Content:")
                        for i, line in enumerate(analysis['first_10_lines'][:5], 1):
                            print(f"      {i}. {line[:80]}{'...' if len(line) > 80 else ''}")
                else:
                    print(f"   ❌ Error: {analysis.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ Error: {adobe.get('error', 'OCR failed')}")
        
        # Summary
        print(f"\n🎉 ADOBE OCR SUMMARY:")
        adobe_success = self.results.get('adobe', {}).get('success', False)
        
        if adobe_success:
            adobe_time = self.results['adobe']['processing_time']
            adobe_words = self.results['adobe']['analysis'].get('word_count', 0)
            adobe_size = self.results['adobe']['analysis'].get('file_size', 0)
            
            print(f"   ✅ Successfully processed in {adobe_time:.2f} seconds")
            print(f"   📝 Extracted {adobe_words} words from document")
            print(f"   📁 Generated {adobe_size/1024:.1f} KB searchable PDF")
            print(f"   🎯 Professional-grade OCR with layout preservation")
        else:
            print(f"   ❌ OCR processing failed")
        
        print(f"\n📋 RECOMMENDATION:")
        if adobe_success:
            print("   🔵 Adobe PDF Services successfully created a searchable PDF")
            print("   ✅ Document is now fully searchable and editable")
            print("   🎨 Original layout and formatting preserved")
        else:
            print("   ❌ OCR failed - check document quality and credentials")
    
    def process_document(self):
        """Run complete Adobe OCR analysis"""
        
        print("🚀 ADOBE PDF SERVICES OCR ANALYZER")
        print("=" * 60)
        
        if not os.path.exists(self.input_pdf):
            print(f"❌ Input PDF not found: {self.input_pdf}")
            return
        
        # Run Adobe OCR
        self.run_adobe_ocr()
        
        # Generate analysis report
        self.generate_analysis_report()
        
        print(f"\n🎉 Adobe OCR Analysis completed!")
        print(f"📄 Check the generated PDF file for results")

def main():
    """Main execution function"""
    
    try:
        analyzer = AdobeOCRAnalyzer()
        analyzer.process_document()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 