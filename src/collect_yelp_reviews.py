#!/usr/bin/env python3
"""
Command-line script to collect reviews from local Yelp dataset files.
"""

import argparse
import logging
import sys
from pathlib import Path

from data_collection.yelp_dataset import YelpDatasetCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect reviews from local Yelp dataset files'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Directory containing Yelp dataset files (yelp_academic_dataset_*.json or .json.gz)'
    )
    
    parser.add_argument(
        '--max-reviews',
        type=int,
        default=500,
        help='Maximum number of reviews to collect (default: 500)'
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

def validate_dataset_dir(dataset_dir: Path) -> bool:
    """Validate that the dataset directory contains required files."""
    required_files = [
        'yelp_academic_dataset_review.json',
        'yelp_academic_dataset_review.json.gz'
    ]
    
    # Check if at least one of the review files exists
    has_review_file = any((dataset_dir / f).exists() for f in required_files)
    
    if not has_review_file:
        logger.error(f"No review file found in {dataset_dir}")
        logger.info("Expected one of:")
        for f in required_files:
            logger.info(f"  - {f}")
        return False
    
    return True

def main():
    """Main function to collect reviews from Yelp dataset files."""
    args = parse_args()
    
    # Validate dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    if not validate_dataset_dir(dataset_dir):
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = YelpDatasetCollector(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir)
    )
    
    try:
        logger.info("Starting Yelp review collection")
        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Maximum reviews to collect: {args.max_reviews}")
        logger.info(f"Minimum review length: {args.min_length} characters")
        
        collector.collect_reviews(
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