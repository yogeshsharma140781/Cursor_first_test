#!/usr/bin/env python3
"""
Extract 500 non-English textual reviews from collected Amazon review files and save to a CSV.
"""

import os
import glob
import pandas as pd
from langdetect import detect
from tqdm import tqdm

DATA_DIR = 'data/reviews'
OUTPUT_CSV = 'data/500_non_english_reviews.csv'
REQUIRED_COUNT = 500


def is_non_english(text):
    try:
        lang = detect(text)
        return lang != 'en'
    except Exception:
        return False

def main():
    all_reviews = []
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    for file in tqdm(files, desc='Processing files'):
        try:
            df = pd.read_csv(file)
            if 'text' in df.columns:
                texts = df['text'].dropna().astype(str).tolist()
            elif 'review_text' in df.columns:
                texts = df['review_text'].dropna().astype(str).tolist()
            else:
                continue
            for text in texts:
                if is_non_english(text):
                    all_reviews.append(text)
                    if len(all_reviews) >= REQUIRED_COUNT:
                        break
            if len(all_reviews) >= REQUIRED_COUNT:
                break
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Truncate to exactly 500
    all_reviews = all_reviews[:REQUIRED_COUNT]
    
    # Save to CSV
    out_df = pd.DataFrame({'review_text': all_reviews})
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Saved {len(all_reviews)} non-English reviews to {OUTPUT_CSV}")

if __name__ == '__main__':
    main() 