import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
async def translate(req: TranslationRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    try:
        prompt = f"Analyze the following text and identify its domain/context (e.g., Legal, Medical, Education, Finance, Technical, etc.). Then translate it from {req.source_lang} to {req.target_lang}, taking into account that this is a General text. Output only the translation (do not prefix 'Domain:' or 'Translation:').\n\nText to analyze and translate:\n{req.text}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            translation = result["choices"][0]["message"]["content"]
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}") 