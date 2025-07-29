import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_translator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepseekTranslator:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        """Initialize the Deepseek translator with the specified model."""
        logger.info(f"Initializing Deepseek translator with model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def translate_text(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate a single text using Deepseek."""
        try:
            prompt = f"""Translate the following text from {source_lang} to {target_lang}. 
            Only provide the translation, no explanations or additional text:
            
            {text}
            
            Translation:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the translation part (after "Translation:")
            translation = translation.split("Translation:")[-1].strip()
            
            return translation
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return ""

    def translate_batch(self, texts: List[str], source_lang: str = "auto", target_lang: str = "en") -> List[str]:
        """Translate a batch of texts."""
        translations = []
        for text in tqdm(texts, desc="Translating batch"):
            translation = self.translate_text(text, source_lang, target_lang)
            translations.append(translation)
        return translations

def load_reviews(file_path: str) -> List[Dict[str, Any]]:
    """Load reviews from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading reviews: {str(e)}")
        return []

def save_translations(reviews: List[Dict[str, Any]], output_file: str):
    """Save translations to CSV file."""
    try:
        df = pd.DataFrame(reviews)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Translations saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving translations: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize translator
    translator = DeepseekTranslator()
    
    # File paths
    input_file = "data/reviews/yelp_dataset_20250602_173302.json"  # Updated path
    output_file = "data/reviews/reviews_with_deepseek.csv"
    
    # Load reviews
    logger.info("Loading reviews...")
    reviews = load_reviews(input_file)
    if not reviews:
        logger.error("No reviews loaded. Exiting.")
        return
    
    # Process reviews in batches
    batch_size = 5  # Adjust based on your GPU memory
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(reviews) + batch_size - 1)//batch_size}")
        
        # Get texts to translate
        texts_to_translate = [review.get('text', '') for review in batch]
        
        # Translate batch
        translations = translator.translate_batch(texts_to_translate)
        
        # Update reviews with translations
        for review, translation in zip(batch, translations):
            review['Deepseek_Translate'] = translation
    
    # Save results
    save_translations(reviews, output_file)
    logger.info("Translation process completed successfully")

if __name__ == "__main__":
    main() 