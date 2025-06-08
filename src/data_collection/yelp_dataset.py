#!/usr/bin/env python3
"""
Yelp Open Dataset Collector
Collects reviews from the local Yelp Open Dataset files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from langdetect import detect
import tarfile
import gzip
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YelpDatasetCollector:
    """Collects reviews from the local Yelp Open Dataset files."""
    
    def __init__(self, dataset_dir: str, output_dir: str = 'data/reviews'):
        """
        Initialize the Yelp Dataset Collector.
        
        Args:
            dataset_dir: Directory containing the Yelp dataset files
            output_dir: Directory to save collected reviews
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected dataset files
        self.review_file = self.dataset_dir / 'yelp_academic_dataset_review.json'
        self.business_file = self.dataset_dir / 'yelp_academic_dataset_business.json'
        
        # Check if files exist
        if not self.review_file.exists():
            # Check for .gz version
            self.review_file = self.dataset_dir / 'yelp_academic_dataset_review.json.gz'
            if not self.review_file.exists():
                raise FileNotFoundError(f"Review file not found in {dataset_dir}")
    
    def is_non_english(self, text: str) -> bool:
        """Detect if the text is not in English."""
        try:
            detected = detect(text)
            return detected != 'en'
        except:
            return False
    
    def process_reviews(
        self,
        max_reviews: int = 500,
        min_review_length: int = 50
    ) -> List[Dict]:
        """
        Process reviews from the Yelp dataset files.
        
        Args:
            max_reviews: Maximum number of reviews to collect
            min_review_length: Minimum length of review text in characters
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        logger.info("Processing Yelp reviews...")
        
        # Determine if we need to decompress the file
        is_gzipped = self.review_file.suffix == '.gz'
        
        try:
            # Open the file (gzipped or plain)
            open_func = gzip.open if is_gzipped else open
            mode = 'rt' if is_gzipped else 'r'
            
            with open_func(self.review_file, mode, encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing reviews"):
                    if len(reviews) >= max_reviews:
                        break
                        
                    try:
                        review = json.loads(line)
                        review_text = review.get('text', '').strip()
                        
                        if len(review_text) < min_review_length:
                            continue
                            
                        if not self.is_non_english(review_text):
                            continue
                            
                        # Get business info if available
                        business_id = review.get('business_id', '')
                        business_name = ''
                        business_city = ''
                        business_country = ''
                        
                        if self.business_file.exists():
                            try:
                                with open(self.business_file, 'r', encoding='utf-8') as bf:
                                    for bline in bf:
                                        business = json.loads(bline)
                                        if business.get('business_id') == business_id:
                                            business_name = business.get('name', '')
                                            business_city = business.get('city', '')
                                            business_country = business.get('state', '')  # Yelp uses state as country code
                                            break
                            except Exception as e:
                                logger.warning(f"Error reading business info: {str(e)}")
                        
                        review_data = {
                            'review_id': review.get('review_id', ''),
                            'business_id': business_id,
                            'business_name': business_name,
                            'business_city': business_city,
                            'business_country': business_country,
                            'text': review_text,
                            'date': review.get('date', ''),
                            'stars': review.get('stars', 0),
                            'language': detect(review_text),
                            'useful': review.get('useful', 0),
                            'funny': review.get('funny', 0),
                            'cool': review.get('cool', 0)
                        }
                        reviews.append(review_data)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing review: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading dataset: {str(e)}")
            
        return reviews
    
    def save_reviews(self, reviews: List[Dict]):
        """Save collected reviews to CSV and JSON files."""
        if not reviews:
            return
            
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"yelp_dataset_{timestamp}"
        
        # Save as CSV
        df = pd.DataFrame(reviews)
        csv_path = self.output_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(reviews)} reviews to {csv_path} and {json_path}")
        
        # Print language distribution
        lang_dist = df['language'].value_counts()
        logger.info("\nLanguage distribution:")
        for lang, count in lang_dist.items():
            logger.info(f"{lang}: {count} reviews")
        
        # Print country distribution
        country_dist = df['business_country'].value_counts()
        logger.info("\nCountry distribution:")
        for country, count in country_dist.items():
            logger.info(f"{country}: {count} reviews")
    
    def collect_reviews(
        self,
        max_reviews: int = 500,
        min_review_length: int = 50
    ):
        """
        Collect reviews from the Yelp dataset.
        
        Args:
            max_reviews: Maximum number of reviews to collect
            min_review_length: Minimum length of review text in characters
        """
        try:
            reviews = self.process_reviews(
                max_reviews=max_reviews,
                min_review_length=min_review_length
            )
            self.save_reviews(reviews)
        except Exception as e:
            logger.error(f"Error during review collection: {str(e)}")

def main():
    """Example usage of the Yelp Dataset Collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect reviews from Yelp dataset')
    parser.add_argument('--dataset-dir', required=True, help='Directory containing Yelp dataset files')
    parser.add_argument('--output-dir', default='data/reviews', help='Directory to save collected reviews')
    parser.add_argument('--max-reviews', type=int, default=500, help='Maximum number of reviews to collect')
    parser.add_argument('--min-length', type=int, default=50, help='Minimum review length in characters')
    
    args = parser.parse_args()
    
    collector = YelpDatasetCollector(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    try:
        collector.collect_reviews(
            max_reviews=args.max_reviews,
            min_review_length=args.min_length
        )
    except KeyboardInterrupt:
        logger.info("Review collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during review collection: {str(e)}")

if __name__ == '__main__':
    main() 