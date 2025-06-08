#!/usr/bin/env python3
"""
Command-line script to collect reviews from the Amazon Review Dataset (2018).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from data_collection.amazon_dataset import AmazonDatasetCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect reviews from the Amazon Review Dataset (2018)'
    )
    
    parser.add_argument(
        '--countries',
        nargs='+',
        default=['es', 'fr', 'de', 'it', 'jp'],
        help='List of country codes to collect reviews from (default: es fr de it jp)'
    )
    
    parser.add_argument(
        '--max-reviews',
        type=int,
        default=500,
        help='Maximum number of reviews to collect per country (default: 500)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum length of review text in characters (default: 50)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/reviews',
        help='Directory to save collected reviews (default: data/reviews)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/cache',
        help='Directory to cache downloaded datasets (default: data/cache)'
    )
    
    return parser.parse_args()

def validate_countries(countries: List[str]) -> bool:
    """Validate country codes against supported datasets."""
    valid_countries = set(AmazonDatasetCollector.SUPPORTED_LANGUAGES)
    invalid_countries = [c for c in countries if c.lower() not in valid_countries]
    
    if invalid_countries:
        logger.error(f"Unsupported country codes: {', '.join(invalid_countries)}")
        logger.info(f"Supported country codes: {', '.join(sorted(valid_countries))}")
        return False
    return True

def main():
    """Main function to collect reviews from the dataset."""
    args = parse_args()
    
    # Validate country codes
    if not validate_countries(args.countries):
        sys.exit(1)
    
    # Create output and cache directories
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = AmazonDatasetCollector(
        output_dir=str(output_dir)
    )
    
    try:
        logger.info(f"Starting review collection for {len(args.countries)} countries")
        logger.info(f"Target countries: {', '.join(args.countries)}")
        logger.info(f"Maximum reviews per country: {args.max_reviews}")
        logger.info(f"Minimum review length: {args.min_length} characters")
        
        collector.collect_reviews(
            country_codes=args.countries,
            max_reviews=args.max_reviews,
            min_review_length=args.min_length
        )
        
        logger.info("Review collection completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\nReview collection interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during review collection: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 