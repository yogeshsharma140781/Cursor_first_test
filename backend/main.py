from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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

@app.post("/translate")
async def translate(req: TranslateRequest):
    prompt = (
        f"Analyze the following text and identify its domain/context (e.g., Legal, Medical, Education, Finance, Technical, etc.). "
        f"Then translate it from {req.source_lang} to {req.target_lang}, taking into account that this is a General text. "
        "Output only the translation (do not prefix 'Domain:' or 'Translation:').\n\n"
        f"Text to analyze and translate:\n{req.text}"
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
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
        result = response.json()
        translation = result["choices"][0]["message"]["content"].strip()
    return {"translation": translation} 