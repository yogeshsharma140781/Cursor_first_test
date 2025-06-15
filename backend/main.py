from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import asyncio
from functools import lru_cache
import hashlib

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Create a persistent HTTP client for connection pooling
client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

# Simple in-memory cache for translations
translation_cache = {}
CACHE_SIZE = 1000

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

def get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a cache key for the translation request."""
    content = f"{text}|{source_lang}|{target_lang}"
    return hashlib.md5(content.encode()).hexdigest()

@app.post("/translate")
async def translate(req: TranslateRequest):
    # Check cache first
    cache_key = get_cache_key(req.text, req.source_lang, req.target_lang)
    if cache_key in translation_cache:
        return {"translation": translation_cache[cache_key]}
    
    # Use faster gpt-4o-mini instead of gpt-4
    prompt = (
        f"Translate the following text from {req.source_lang} to {req.target_lang}. "
        "Output only the translation without any additional text or explanations.\n\n"
        f"Text: {req.text}"
    )
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",  # Much faster than gpt-4
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    response = await client.post(url, headers=headers, json=data)
    result = response.json()
    translation = result["choices"][0]["message"]["content"].strip()
    
    # Cache the result (with size limit)
    if len(translation_cache) >= CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(translation_cache))
        del translation_cache[oldest_key]
    
    translation_cache[cache_key] = translation
    
    return {"translation": translation}

@app.on_event("startup")
async def startup_event():
    """Warm up the connection to reduce cold start latency."""
    try:
        # Make a simple request to warm up connections
        await client.get("https://api.openai.com/v1/models", 
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
    except:
        pass  # Ignore errors during warmup

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    await client.aclose() 