#!/usr/bin/env python3
"""
Amazon Review Collector
Collects product reviews from Amazon's public review pages.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonReviewCollector:
    """Collects product reviews from Amazon's public review pages."""
    
    # Amazon domains for different countries
    AMAZON_DOMAINS = {
        'es': 'amazon.es',      # Spain
        'fr': 'amazon.fr',      # France
        'de': 'amazon.de',      # Germany
        'it': 'amazon.it',      # Italy
        'jp': 'amazon.co.jp',   # Japan
        'in': 'amazon.in',      # India
        'ca': 'amazon.ca',      # Canada
        'mx': 'amazon.com.mx',  # Mexico
        'br': 'amazon.com.br',  # Brazil
    }
    
    def __init__(self, output_dir: str = 'data/reviews'):
        """Initialize the Amazon Review Collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ua = UserAgent()
        self.session = requests.Session()
    
    def get_headers(self) -> Dict[str, str]:
        """Generate random headers for requests."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
    
    def get_product_url(self, asin: str, country_code: str) -> str:
        """Generate Amazon product URL for a specific country."""
        domain = self.AMAZON_DOMAINS.get(country_code.lower())
        if not domain:
            raise ValueError(f"Unsupported country code: {country_code}")
        return f"https://www.{domain}/dp/{asin}/"
    
    def get_reviews_url(self, asin: str, country_code: str, page: int = 1) -> str:
        """Generate Amazon reviews URL for a specific product and page."""
        domain = self.AMAZON_DOMAINS.get(country_code.lower())
        if not domain:
            raise ValueError(f"Unsupported country code: {country_code}")
        return f"https://www.{domain}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={page}"
    
    def get_reviews(
        self,
        asin: str,
        country_code: str,
        max_pages: int = 5,
        delay: float = 2.0
    ) -> List[Dict]:
        """
        Collect reviews for a specific product from Amazon.
        
        Args:
            asin: Amazon product ASIN
            country_code: Country code for Amazon domain
            max_pages: Maximum number of review pages to collect
            delay: Delay between requests in seconds
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        for page in tqdm(range(1, max_pages + 1), desc=f"Collecting reviews from {country_code}"):
            try:
                url = self.get_reviews_url(asin, country_code, page)
                response = self.session.get(url, headers=self.get_headers(), timeout=15)
                
                # Check for common error responses
                if response.status_code == 404:
                    logger.warning(f"Product {asin} not found in {country_code}")
                    break
                elif response.status_code == 503:
                    logger.warning(f"Amazon is blocking requests. Waiting longer before retry...")
                    time.sleep(delay * 3)  # Wait longer when blocked
                    continue
                elif response.status_code != 200:
                    logger.error(f"Unexpected status code {response.status_code} for {url}")
                    break
                
                # Check for CAPTCHA or bot detection
                if "Enter the characters you see below" in response.text or "Type the characters you see in this image" in response.text:
                    logger.error("Amazon CAPTCHA detected. Try again later with a longer delay.")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                review_elements = soup.find_all('div', {'data-hook': 'review'})
                
                if not review_elements:
                    if "No customer reviews" in response.text:
                        logger.info(f"No reviews found for {asin} in {country_code}")
                    else:
                        logger.warning(f"No reviews found on page {page} for {asin} in {country_code}")
                    break
                
                for review in review_elements:
                    try:
                        # Extract review text
                        review_text = review.find('span', {'data-hook': 'review-body'})
                        if not review_text:
                            continue
                        review_text = review_text.text.strip()
                        
                        # Skip if review is too short
                        if len(review_text.split()) < 5:
                            continue
                        
                        # Get review date
                        date_element = review.find('span', {'data-hook': 'review-date'})
                        review_date = date_element.text.strip() if date_element else ''
                        
                        review_data = {
                            'asin': asin,
                            'country_code': country_code,
                            'text': review_text,
                            'date': review_date,
                        }
                        reviews.append(review_data)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing review: {str(e)}")
                        continue
                
                # Random delay between requests (longer for subsequent pages)
                time.sleep(delay + random.uniform(2, 4) * (page / 2))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching reviews page {page}: {str(e)}")
                time.sleep(delay * 2)  # Wait longer on network errors
                continue
            except Exception as e:
                logger.error(f"Unexpected error on page {page}: {str(e)}")
                break
        
        return reviews
    
    def save_reviews(self, reviews: List[Dict], asin: str, country_code: str):
        """Save collected reviews to CSV and JSON files."""
        if not reviews:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{asin}_{country_code}_{timestamp}"
        
        # Save as CSV
        df = pd.DataFrame(reviews)
        csv_path = self.output_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(reviews)} reviews to {csv_path} and {json_path}")
    
    def collect_multiple_products(
        self,
        asins: List[str],
        country_codes: List[str],
        max_pages: int = 5,
        delay: float = 2.0
    ):
        """
        Collect reviews for multiple products from multiple countries.
        
        Args:
            asins: List of Amazon product ASINs
            country_codes: List of country codes
            max_pages: Maximum number of review pages to collect per product
            delay: Delay between requests in seconds
        """
        for asin in asins:
            for country_code in country_codes:
                try:
                    reviews = self.get_reviews(asin, country_code, max_pages, delay)
                    self.save_reviews(reviews, asin, country_code)
                except Exception as e:
                    logger.error(f"Error processing {asin} from {country_code}: {str(e)}")
                time.sleep(delay * 2)  # Additional delay between products

def main():
    """Example usage of the Amazon Review Collector."""
    # Example product ASINs
    asins = [
        'B08N5KWB9H',  # iPhone 12
        'B07ZPKN6YR',  # AirPods Pro
        'B08L5TNJHG',  # Samsung Galaxy S21
    ]
    
    # Countries to collect reviews from
    countries = ['es', 'fr', 'de', 'it', 'jp']
    
    collector = AmazonReviewCollector()
    try:
        collector.collect_multiple_products(asins, countries)
    except KeyboardInterrupt:
        logger.info("Review collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during review collection: {str(e)}")

if __name__ == '__main__':
    main() 