#!/usr/bin/env python3
"""
Command-line script to collect Amazon reviews from public review pages.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from data_collection.amazon_reviews import AmazonReviewCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect Amazon reviews from public review pages'
    )
    
    parser.add_argument(
        '--asins',
        nargs='+',
        required=True,
        help='List of Amazon product ASINs to collect reviews for'
    )
    
    parser.add_argument(
        '--countries',
        nargs='+',
        default=['es', 'fr', 'de', 'it', 'jp'],
        help='List of country codes to collect reviews from (default: es fr de it jp)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=5,
        help='Maximum number of review pages to collect per product (default: 5)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/reviews',
        help='Directory to save collected reviews (default: data/reviews)'
    )
    
    return parser.parse_args()

def validate_countries(countries: List[str]) -> bool:
    """Validate country codes against supported Amazon domains."""
    valid_countries = set(AmazonReviewCollector.AMAZON_DOMAINS.keys())
    invalid_countries = [c for c in countries if c.lower() not in valid_countries]
    
    if invalid_countries:
        logger.error(f"Unsupported country codes: {', '.join(invalid_countries)}")
        logger.info(f"Supported country codes: {', '.join(sorted(valid_countries))}")
        return False
    return True

def main():
    """Main function to collect Amazon reviews."""
    args = parse_args()
    
    # Validate country codes
    if not validate_countries(args.countries):
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = AmazonReviewCollector(output_dir=str(output_dir))
    
    try:
        logger.info(f"Starting review collection for {len(args.asins)} products")
        logger.info(f"Target countries: {', '.join(args.countries)}")
        logger.info(f"Maximum pages per product: {args.max_pages}")
        logger.info(f"Delay between requests: {args.delay} seconds")
        
        collector.collect_multiple_products(
            asins=args.asins,
            country_codes=args.countries,
            max_pages=args.max_pages,
            delay=args.delay
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