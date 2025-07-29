#!/usr/bin/env python3

import re

def _should_preserve_original(text: str) -> bool:
    """
    Determine if text should be preserved in original language instead of translated
    Returns True for addresses, postal codes, technical terms, URLs, etc.
    """
    text_clean = text.strip()
    
    # Special case: phrases with colons that should be translated (check this first)
    if ':' in text_clean:
        colon_phrases_to_translate = ['nationality:', 'born on:', 'date:', 'subject:']
        text_lower = text_clean.lower()
        if any(phrase in text_lower for phrase in colon_phrases_to_translate):
            return False
    
    # Preserve pure technical codes and numbers (no mixed text)
    if len(text_clean) < 15 and re.match(r'^[A-Z0-9\-\s]+$', text_clean) and not re.search(r'[a-z]', text_clean):
        return True
    
    # Preserve postal codes (Dutch format like "9560 AA")
    if re.match(r'^\d{4}\s+[A-Z]{2}$', text_clean):
        return True
    
    # Preserve addresses (contain numbers and specific address words)
    address_indicators = ['straat', 'laan', 'weg', 'plein', 'park', 'kade', 'gracht', 'singel']
    if any(indicator in text_clean.lower() for indicator in address_indicators) and re.search(r'\d+', text_clean):
        return True
        
    # Preserve URLs and email addresses
    if re.search(r'www\.|http[s]?://|@.*\.|\.nl|\.com|\.org', text_clean.lower()):
        return True
    
    # Preserve phone numbers
    if re.match(r'^[T\s]*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}$', text_clean):
        return True
        
    # Preserve case numbers and reference codes (pure alphanumeric)
    if re.match(r'^[A-Z]\d+-\d+$', text_clean) or re.match(r'^\d{10,}$', text_clean):
        return True
        
    # Preserve specific technical terms that should stay in English
    technical_terms = ['V-number', 'RVN', 'NAT', 'ZW', 'Team']
    if any(term.lower() in text_clean.lower() for term in technical_terms):
        return True
    
    # Preserve names that are likely proper nouns (mixed case, not common words)
    # BUT exclude common phrases that should be translated
    if (len(text_clean.split()) <= 3 and 
        any(word[0].isupper() and word[1:].islower() for word in text_clean.split() if len(word) > 1)):
        
        # Check if it's not a common Dutch word that should be translated
        common_dutch_words = ['De', 'Het', 'Een', 'Van', 'Voor', 'Met', 'Door', 'Naar', 'Aan', 'In', 'Op', 'Bij']
        if not any(word in common_dutch_words for word in text_clean.split()):
            
            # Additional check: exclude common phrases that should be translated
            common_phrases_to_translate = [
                'case number', 'case numbers',
                'nationality', 'nationalities', 
                'born on', 'born in',
                'postbus', 'post box', 'post office',
                'date', 'dates',
                'subject', 'subjects',
                'team', 'teams'
            ]
            
            text_lower = text_clean.lower()
            if not any(phrase in text_lower for phrase in common_phrases_to_translate):
                return True
    
    return False

# Test cases
test_texts = [
    "Case number",
    "born on 14 July 1981",
    "nationality: Indian", 
    "V-number",
    "2850241598",
    "Z1-186720992110",
    "IJburglaan 816",
    "1087 EM AMSTERDAM",
    "www.ind.nl",
    "T 088 043 04 30",
    "9560 AA TER APEL",
    "Postbus 3",
    "RVN NAT ZW Team 05",
    "Date",
    "Subject"
]

print("Testing preservation logic:")
print("=" * 50)

for text in test_texts:
    should_preserve = _should_preserve_original(text)
    print(f"'{text}' -> {'PRESERVE' if should_preserve else 'TRANSLATE'}") 