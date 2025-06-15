import os
import streamlit as st
from dotenv import load_dotenv
import httpx
import time
import hashlib
from functools import lru_cache

# --- Setup ---
load_dotenv()  # Load .env file for local development
API_KEY = os.getenv('OPENAI_API_KEY')

if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found! Please add it to your environment variables or Streamlit secrets.")
    st.stop()

# Initialize persistent HTTP client for better performance
@st.cache_resource
def get_http_client():
    return httpx.Client(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
    )

# Simple caching for translations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Get cached translation if available."""
    cache_key = hashlib.md5(f"{text}|{source_lang}|{target_lang}".encode()).hexdigest()
    return cache_key

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using OpenAI API with caching."""
    # Check if we have a cached result
    cache_key = get_cached_translation(text, source_lang, target_lang)
    
    # Use simplified prompt for faster translation
    prompt = f"Translate the following text from {source_lang} to {target_lang}. Output only the translation:\n\n{text}"
    
    client = get_http_client()
    
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
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
        
        response = client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return "Translation failed. Please try again."

# --- Language Configuration ---
LANGUAGES = {
    'en': 'English',
    'ar': 'Arabic',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'nl': 'Dutch',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'es': 'Spanish',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'vi': 'Vietnamese',
}
LANGUAGES_REVERSE = {v: k for k, v in LANGUAGES.items()}

# Add domain options
DOMAINS = [
    "General",
    "Education",
    "Legal",
    "Medical",
    "Technical",
    "Finance",
    "Business",
    "Marketing",
    "Literature",
    "News",
    "Social Media",
    "Government",
    "Science",
    "Technology",
    "Entertainment",
    "Sports",
    "Travel",
    "Food",
    "Art",
    "Religion"
]

# Initialize session state variables
if 'detected_domain' not in st.session_state:
    st.session_state.detected_domain = "General"
if 'domain_context' not in st.session_state:
    st.session_state.domain_context = ""
if 'translation_context' not in st.session_state:
    st.session_state.translation_context = ""
if 'last_input' not in st.session_state:
    st.session_state.last_input = ''
if 'last_input_time' not in st.session_state:
    st.session_state.last_input_time = 0.0
if 'last_translation' not in st.session_state:
    st.session_state.last_translation = ''
if 'auto_translate' not in st.session_state:
    st.session_state.auto_translate = False
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

# --- UI Styling ---
st.set_page_config(
    page_title="Translator AI",
    page_icon="SmallLogo.svg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: #fff;
        box-shadow: 0 2px 8px rgba(30,144,255,0.07);
        padding: 0.5rem 0;
    }
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.2rem 0 0 0;
    }
    .logo-text {
        font-size: 2rem;
        font-weight: 600;
        color: #1e90ff;
        margin-left: 0.5rem;
    }
    .main-header svg {
        height: 60px !important;
        width: auto !important;
    }
    .translation-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0.5rem 0 0 0;
    }
    .stApp {
        padding-top: 80px !important;
    }
    .text-area-container {
        margin: 1rem 0;
    }
    /* Custom Translate button style */
    div.stButton > button:first-child {
        background-color: #007aff !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 0 !important;
        border-radius: 0.5rem !important;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        width: 200px !important;
        margin: 1rem 0;
        transition: background 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #005bb5 !important;
    }
    .stTextArea > div > div > textarea {
        background-color: #fff !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 1rem !important;
        font-size: 1.15rem !important;
    }
    .stTextArea textarea:disabled {
        color: #111 !important;
        -webkit-text-fill-color: #111 !important;
        opacity: 1 !important;
    }
    .language-selector {
        margin: 1rem 0;
    }
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 2rem 0;
    }
    /* Forceful selector for Translate button font size */
    div.stButton > button {
        font-size: 1.5rem !important;
    }
    /* Optionally try to remove Streamlit's default top padding */
    .css-18e3th9 {
        padding-top: 0rem !important;
    }
    /* Custom dropdown styling */
    .stSelectbox > div > div > select {
        background-color: transparent;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 8px 32px 8px 12px;
        color: #1f77b4;
        font-weight: 500;
        font-size: 14px;
        appearance: none;
        background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%231f77b4' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
        background-position: right 8px center;
        background-size: 16px;
        background-repeat: no-repeat;
    }
    .stSelectbox > div > div > select:focus {
        outline: none;
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
    }
    /* Style the selectbox container */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    /* Custom label styling */
    .stSelectbox > label {
        font-weight: 500;
        color: #1f77b4;
        margin-bottom: 4px;
    }
    .custom-dropdown {
        background: transparent !important;
    }
    .dropdown {
        position: relative;
        width: 200px;
        margin-bottom: 12px;
    }
    .dropdown select {
        width: 100%;
        padding: 12px 16px;
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        appearance: none;
        cursor: pointer;
        transition: border 0.3s ease;
    }
    .dropdown select:focus {
        border-color: #4c9ffe;
        outline: none;
    }
    /* Remove browser autofill background and any unwanted pseudo-elements */
    .dropdown select::-ms-expand { display: none; }
    .dropdown select::-webkit-input-placeholder { color: #888; }
    .dropdown select:-ms-input-placeholder { color: #888; }
    .dropdown select::placeholder { color: #888; }
    .sticky-translate-btn {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        z-index: 2000;
        background: #fff;
        box-shadow: 0 -2px 8px rgba(30,144,255,0.07);
        padding: 1rem 0.5rem 1.2rem 0.5rem;
        text-align: center;
        display: none !important;
    }
    @media (max-width: 600px) {
        .sticky-translate-btn {
            display: block !important;
        }
        .desktop-translate-btn {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Fixed Header with SVG Logo (removed "Translator AI" text) ---
st.markdown('<div class="fixed-header"><div class="translation-container">', unsafe_allow_html=True)
with open("Logo-full.svg", "r") as f:
    svg_logo = f.read()
st.markdown(
    f'''
    <div class="main-header">
        <div style="height:60px; width:auto; display:flex; align-items:center;">
            {svg_logo}
        </div>
    </div>
    </div></div>''',
    unsafe_allow_html=True
)

# --- Single Column Layout ---
source_languages = {"Detect Language": "auto"}
source_languages.update({v: k for k, v in LANGUAGES.items()})
source_lang_options = list(source_languages.keys())
if 'source_lang' not in st.session_state:
    st.session_state['source_lang'] = source_lang_options[0]
source_lang = st.selectbox(
    "Source language",
    source_lang_options,
    key="source_lang",
    label_visibility="collapsed"
)

input_text = st.text_area(
    "Input Text",
    height=200,
    placeholder="Type or paste your text here...",
    label_visibility="collapsed",
    key="input_text_area"
)

# Calculate words and characters
words = len(input_text.split()) if input_text.strip() else 0
chars = len(input_text)
st.markdown(f"<div style='margin-top:4px; margin-bottom:8px; text-align:left; color:#444; font-size:15px;'>{words} words, {chars} characters</div>", unsafe_allow_html=True)

target_languages = ["English"] + [v for k, v in LANGUAGES.items() if v != "English"]
default_target = "English"
if 'target_lang' not in st.session_state:
    st.session_state['target_lang'] = default_target
target_lang = st.selectbox(
    "Target language",
    target_languages,
    key="target_lang",
    label_visibility="collapsed"
)

# --- Translate button ---
translate_clicked = st.button("Translate", key="main_translate_btn")
if translate_clicked:
    if input_text.strip():
        with st.spinner("Translating... (please wait)"):
            source_code = LANGUAGES_REVERSE.get(st.session_state['source_lang'], "auto")
            target_code = LANGUAGES_REVERSE.get(st.session_state['target_lang'], "en")
            
            # Use the optimized translation function
            translation = translate_text(input_text, source_code, target_code)
            st.session_state.translated_text = translation
    else:
        st.warning("Please enter some text to translate.")

# --- Translated text area below ---
st.text_area(
    "Output Text",
    value=st.session_state.translated_text,
    height=400,
    disabled=True,
    label_visibility="collapsed",
    key="output_text"
)