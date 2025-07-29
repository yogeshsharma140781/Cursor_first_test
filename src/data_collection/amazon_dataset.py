#!/usr/bin/env python3
"""
Amazon Review Dataset Collector
Collects reviews from the Amazon Review Dataset (2018).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import requests
from langdetect import detect
import gzip
import shutil
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonDatasetCollector:
    """Collects reviews from the Amazon Review Dataset (2018) using Hugging Face datasets."""
    
    SUPPORTED_LANGUAGES = ['es', 'fr', 'de', 'it', 'jp']
    
    def __init__(self, output_dir: str = 'data/reviews'):
        """Initialize the Amazon Dataset Collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def is_non_english(self, text: str, lang_code: str) -> bool:
        """Detect if the text is not in English (or not in the target language)."""
        try:
            detected = detect(text)
            return detected != 'en' and detected == lang_code
        except:
            return False
    
    def process_reviews(
        self,
        country_code: str,
        max_reviews: int = 500,
        min_review_length: int = 50
    ) -> List[Dict]:
        """
        Process reviews from the Hugging Face dataset for a specific country.
        
        Args:
            country_code: Language code for the dataset
            max_reviews: Maximum number of reviews to collect
            min_review_length: Minimum length of review text in characters
            
        Returns:
            List of review dictionaries
        """
        if country_code not in self.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported country code: {country_code}")
            return []
        
        logger.info(f"Loading dataset for {country_code} from Hugging Face...")
        try:
            dataset = load_dataset('amazon_reviews_multi', country_code, split='train', trust_remote_code=True)
        except Exception as e:
            logger.error(f"Error loading dataset for {country_code}: {str(e)}")
            return []
        
        reviews = []
        for item in tqdm(dataset, desc=f"Processing {country_code} reviews"):
            if len(reviews) >= max_reviews:
                break
            review_text = item.get('review_body', '').strip()
            if len(review_text) < min_review_length:
                continue
            # Optionally, check language
            if not self.is_non_english(review_text, country_code):
                continue
            review_data = {
                'asin': item.get('product_id', ''),
                'country_code': country_code,
                'text': review_text,
                'date': item.get('review_date', ''),
            }
            reviews.append(review_data)
        return reviews
    
    def save_reviews(self, reviews: List[Dict], country_code: str):
        """Save collected reviews to CSV and JSON files."""
        if not reviews:
            return
            
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"amazon_dataset_{country_code}_{timestamp}"
        
        # Save as CSV
        df = pd.DataFrame(reviews)
        csv_path = self.output_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(reviews)} reviews to {csv_path} and {json_path}")
    
    def collect_reviews(
        self,
        country_codes: List[str],
        max_reviews: int = 500,
        min_review_length: int = 50
    ):
        """
        Collect reviews from the dataset for multiple countries.
        
        Args:
            country_codes: List of country codes
            max_reviews: Maximum number of reviews to collect per country
            min_review_length: Minimum length of review text in characters
        """
        for country_code in country_codes:
            try:
                reviews = self.process_reviews(
                    country_code,
                    max_reviews=max_reviews,
                    min_review_length=min_review_length
                )
                self.save_reviews(reviews, country_code)
            except Exception as e:
                logger.error(f"Error processing {country_code}: {str(e)}")

def main():
    """Example usage of the Amazon Dataset Collector."""
    # Countries to collect reviews from
    countries = ['es', 'fr', 'de', 'it', 'jp']
    
    collector = AmazonDatasetCollector()
    try:
        collector.collect_reviews(countries)
    except KeyboardInterrupt:
        logger.info("Review collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during review collection: {str(e)}")

if __name__ == '__main__':
    main() 