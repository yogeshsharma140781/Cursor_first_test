#!/usr/bin/env python3
"""
Trustpilot Review Collector
Collects reviews from Trustpilot using their public API.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from langdetect import detect
import requests
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrustpilotReviewCollector:
    """Collects reviews from Trustpilot using their public API."""
    
    BASE_URL = "https://api.trustpilot.com/v1"
    BUSINESS_UNITS = {
        'us': '4f4c5c9c00006400050d8b2d',  # Example business unit ID for US
        'uk': '4f4c5c9c00006400050d8b2e',  # Example business unit ID for UK
        'de': '4f4c5c9c00006400050d8b2f',  # Example business unit ID for Germany
        'fr': '4f4c5c9c00006400050d8b30',  # Example business unit ID for France
        'es': '4f4c5c9c00006400050d8b31',  # Example business unit ID for Spain
        'it': '4f4c5c9c00006400050d8b32',  # Example business unit ID for Italy
    }
    
    def __init__(self, output_dir: str = 'data/reviews'):
        """Initialize the Trustpilot Review Collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # Add a user agent to identify our application
        self.session.headers.update({
            'User-Agent': 'ReviewCollector/1.0 (Educational Project)'
        })
    
    def get_reviews(
        self,
        business_unit_id: str,
        per_page: int = 100,
        max_reviews: int = 500,
        min_review_length: int = 50
    ) -> List[Dict]:
        """
        Get reviews for a specific business unit.
        
        Args:
            business_unit_id: Trustpilot business unit ID
            per_page: Number of reviews per page
            max_reviews: Maximum number of reviews to collect
            min_review_length: Minimum length of review text in characters
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        page = 1
        
        while len(reviews) < max_reviews:
            try:
                url = f"{self.BASE_URL}/business-units/{business_unit_id}/reviews"
                params = {
                    'perPage': per_page,
                    'page': page,
                    'stars': '1,2,3,4,5',  # Get all ratings
                    'language': 'all'      # Get reviews in all languages
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('reviews'):
                    break
                    
                for review in data['reviews']:
                    if len(reviews) >= max_reviews:
                        break
                        
                    review_text = review.get('text', '').strip()
                    if len(review_text) < min_review_length:
                        continue
                        
                    # Detect language
                    try:
                        language = detect(review_text)
                        if language == 'en':  # Skip English reviews
                            continue
                    except:
                        continue
                        
                    review_data = {
                        'review_id': review.get('id', ''),
                        'business_unit_id': business_unit_id,
                        'text': review_text,
                        'date': review.get('createdAt', ''),
                        'stars': review.get('stars', 0),
                        'language': language,
                        'country': review.get('country', ''),
                        'title': review.get('title', '')
                    }
                    reviews.append(review_data)
                
                page += 1
                # Respect rate limits
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching reviews: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Error processing reviews: {str(e)}")
                break
                
        return reviews
    
    def save_reviews(self, reviews: List[Dict], country_code: str):
        """Save collected reviews to CSV and JSON files."""
        if not reviews:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"trustpilot_{country_code}_{timestamp}"
        
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
    
    def collect_reviews(
        self,
        country_codes: List[str],
        max_reviews: int = 500,
        min_review_length: int = 50
    ):
        """
        Collect reviews from multiple countries.
        
        Args:
            country_codes: List of country codes to collect reviews from
            max_reviews: Maximum number of reviews to collect per country
            min_review_length: Minimum length of review text in characters
        """
        for country_code in country_codes:
            if country_code not in self.BUSINESS_UNITS:
                logger.warning(f"Unsupported country code: {country_code}")
                continue
                
            logger.info(f"Collecting reviews for {country_code}")
            try:
                reviews = self.get_reviews(
                    business_unit_id=self.BUSINESS_UNITS[country_code],
                    max_reviews=max_reviews,
                    min_review_length=min_review_length
                )
                self.save_reviews(reviews, country_code)
            except Exception as e:
                logger.error(f"Error collecting reviews for {country_code}: {str(e)}")

def main():
    """Example usage of the Trustpilot Review Collector."""
    collector = TrustpilotReviewCollector()
    try:
        # Example: collect reviews from Germany, France, and Spain
        collector.collect_reviews(['de', 'fr', 'es'])
    except KeyboardInterrupt:
        logger.info("Review collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during review collection: {str(e)}")

if __name__ == '__main__':
    main() 