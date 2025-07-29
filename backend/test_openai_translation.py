#!/usr/bin/env python3
"""
Test OpenAI translation functionality
"""

import os
try:
    from openai import OpenAI
    NEW_OPENAI = True
except ImportError:
    import openai
    NEW_OPENAI = False

def test_openai_translation():
    """Test OpenAI translation with a simple Dutch text"""
    print("🧪 Testing OpenAI Translation...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found in environment variables")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        # Test translation
        test_text = "We gaan uw kinderopvangtoeslag over 2024 definitief berekenen."
        print(f"📝 Testing translation of: '{test_text}'")
        
        if NEW_OPENAI:
            # Use new OpenAI API (v1.0+)
            client = OpenAI(api_key=api_key)
            print("✅ OpenAI client (v1.0+) initialized successfully")
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the following Dutch text to English. Maintain the meaning and tone. Return only the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": test_text
                    }
                ],
                temperature=0.2,
                max_tokens=200
            )
            translation = response.choices[0].message.content.strip()
        else:
            # Use legacy OpenAI API
            openai.api_key = api_key
            print("✅ OpenAI API key (legacy) set successfully")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the following Dutch text to English. Maintain the meaning and tone. Return only the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": test_text
                    }
                ],
                temperature=0.2,
                max_tokens=200
            )
            translation = response.choices[0].message.content.strip()
        print(f"🌍 Translation result: '{translation}'")
        print("✅ OpenAI translation test successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI translation test failed: {e}")
        return False

if __name__ == "__main__":
    test_openai_translation() 