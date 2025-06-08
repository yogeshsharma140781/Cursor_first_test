#!/usr/bin/env python3
"""
Command-line script to collect reviews from Trustpilot.
"""

import argparse
import logging
import sys
from pathlib import Path

from data_collection.trustpilot_reviews import TrustpilotReviewCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect reviews from Trustpilot'
    )
    
    parser.add_argument(
        '--countries',
        nargs='+',
        default=['de', 'fr', 'es', 'it'],
        help='List of country codes to collect reviews from (default: de fr es it)'
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
    
    return parser.parse_args()

def validate_countries(countries: list) -> bool:
    """Validate country codes against supported countries."""
    valid_countries = set(TrustpilotReviewCollector.BUSINESS_UNITS.keys())
    invalid_countries = [c for c in countries if c.lower() not in valid_countries]
    
    if invalid_countries:
        logger.error(f"Unsupported country codes: {', '.join(invalid_countries)}")
        logger.info(f"Supported country codes: {', '.join(sorted(valid_countries))}")
        return False
    return True

def main():
    """Main function to collect reviews from Trustpilot."""
    args = parse_args()
    
    # Validate country codes
    if not validate_countries(args.countries):
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = TrustpilotReviewCollector(
        output_dir=str(output_dir)
    )
    
    try:
        logger.info("Starting Trustpilot review collection")
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