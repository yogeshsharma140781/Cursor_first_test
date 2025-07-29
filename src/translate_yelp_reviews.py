#!/usr/bin/env python3
"""
Script to translate collected Yelp reviews and save them in a CSV format.
Each row contains: original_text, Google_Translate, OpenAI_Translations
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
import time
import asyncio
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure OpenAI
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewTranslator:
    """Translates reviews from various languages to English using both Google Translate and OpenAI."""
    
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the translator.
        
        Args:
            input_file: Path to the input JSON file containing reviews
            output_file: Path to save the translated CSV file
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.translator = Translator()
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_reviews(self) -> List[Dict]:
        """Load reviews from the input JSON file."""
        logger.info(f"Loading reviews from {self.input_file}")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            # Take only first 100 reviews
            reviews = reviews[:100]
            logger.info(f"Loaded {len(reviews)} reviews (limited to first 100)")
            return reviews
        except Exception as e:
            logger.error(f"Error loading reviews: {str(e)}")
            sys.exit(1)
    
    async def translate_with_openai(self, text: str, src_lang: str) -> str:
        """
        Translate text using OpenAI's API.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            
        Returns:
            Translated text or error message
        """
        try:
            # Skip if already English
            if src_lang == 'en':
                logger.info("Skipping OpenAI translation for English text")
                return text

            logger.info(f"Starting OpenAI translation for {src_lang} text")
            # Add delay to respect rate limits
            await asyncio.sleep(1)  # OpenAI has stricter rate limits

            # Create the prompt for translation
            prompt = f"Translate the following {src_lang} text to English. Only provide the translation, no explanations:\n\n{text}"

            logger.info("Calling OpenAI API...")
            # Call OpenAI API
            try:
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Translate the given text to English."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent translations
                    max_tokens=1000
                )
                translation = response.choices[0].message.content.strip()
                logger.info("OpenAI translation successful")
                return translation
            except Exception as api_error:
                logger.error(f"OpenAI API error: {str(api_error)}")
                raise

        except Exception as e:
            logger.error(f"Error in OpenAI translation: {str(e)}")
            return f"[OpenAI Translation Error: {str(e)}]"

    async def translate_text(self, text: str, src_lang: str) -> Tuple[str, str, str]:
        """
        Translate text from source language to English using both Google Translate and OpenAI.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            
        Returns:
            Tuple of (original_text, google_translated_text, openai_translated_text)
        """
        try:
            # Skip if already English
            if src_lang == 'en':
                logger.info("Text is already in English, skipping translations")
                return text, text, text
                
            logger.info(f"Starting translations for {src_lang} text")
            # Add delay to respect rate limits
            await asyncio.sleep(0.5)
            
            # Translate with Google
            logger.info("Starting Google translation...")
            google_result = await self.translator.translate(text, src=src_lang, dest='en')
            logger.info("Google translation completed")
            
            # Translate with OpenAI
            logger.info("Starting OpenAI translation...")
            openai_translation = await self.translate_with_openai(text, src_lang)
            logger.info("OpenAI translation completed")
            
            return text, google_result.text, openai_translation
            
        except Exception as e:
            logger.error(f"Error in translation process: {str(e)}")
            return text, f"[Translation Error: {str(e)}]", f"[Translation Error: {str(e)}]"
    
    async def translate_reviews(self, reviews: List[Dict]) -> List[Tuple[str, str, str]]:
        """
        Translate all reviews to English using both services.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            List of (original_text, google_translated_text, openai_translated_text) tuples
        """
        translations = []
        logger.info("Translating reviews...")
        
        # Process reviews in smaller batches due to OpenAI rate limits
        batch_size = 5  # Reduced batch size for OpenAI
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            tasks = []
            
            for review in batch:
                original_text = review['text']
                src_lang = review['language']
                task = self.translate_text(original_text, src_lang)
                tasks.append(task)
            
            # Wait for all translations in the batch to complete
            batch_results = await asyncio.gather(*tasks)
            translations.extend(batch_results)
            
            # Update progress bar
            tqdm.write(f"Translated {min(i + batch_size, len(reviews))}/{len(reviews)} reviews")
            
            # Add a longer delay between batches for OpenAI rate limits
            await asyncio.sleep(2)
            
        return translations
    
    def save_translations(self, translations: List[Tuple[str, str, str]]):
        """
        Save translations to CSV file.
        
        Args:
            translations: List of (original_text, google_translated_text, openai_translated_text) tuples
        """
        logger.info(f"Saving translations to {self.output_file}")
        
        # Create DataFrame
        df = pd.DataFrame(translations, columns=['original_text', 'Google_Translate', 'OpenAI_Translations'])
        
        # Save to CSV
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        logger.info(f"Saved {len(translations)} translations")
    
    async def process(self):
        """Process all reviews: load, translate, and save."""
        try:
            # Load reviews
            reviews = self.load_reviews()
            
            # Translate reviews
            translations = await self.translate_reviews(reviews)
            
            # Save translations
            self.save_translations(translations)
            
            logger.info("Translation completed successfully")
            
        except KeyboardInterrupt:
            logger.info("\nTranslation interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            sys.exit(1)

async def main():
    """Main function to translate reviews."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate Yelp reviews to English')
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input JSON file containing reviews'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/reviews/translated_reviews.csv',
        help='Path to save translated CSV file'
    )
    
    args = parser.parse_args()
    
    translator = ReviewTranslator(
        input_file=args.input_file,
        output_file=args.output_file
    )
    
    await translator.process()

if __name__ == '__main__':
    asyncio.run(main()) 