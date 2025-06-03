import os
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
API_KEY = os.getenv('DEEPSEEK_API_KEY')
assert API_KEY, "DEEPSEEK_API_KEY not found in environment!"

# Language codes (Google Translate style, but you can adjust as needed)
LANGUAGES = {
    'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic', 'hy': 'Armenian', 'az': 'Azerbaijani',
    'eu': 'Basque', 'be': 'Belarusian', 'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
    'ceb': 'Cebuano', 'ny': 'Chichewa', 'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
    'eo': 'Esperanto', 'et': 'Estonian', 'tl': 'Filipino', 'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian',
    'gl': 'Galician', 'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gu': 'Gujarati', 'ht': 'Haitian Creole',
    'ha': 'Hausa', 'haw': 'Hawaiian', 'iw': 'Hebrew', 'he': 'Hebrew', 'hi': 'Hindi', 'hmn': 'Hmong', 'hu': 'Hungarian',
    'is': 'Icelandic', 'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian', 'ja': 'Japanese',
    'jw': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh', 'km': 'Khmer', 'ko': 'Korean', 'ku': 'Kurdish (Kurmanji)',
    'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian', 'lt': 'Lithuanian', 'lb': 'Luxembourgish',
    'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mi': 'Maori',
    'mr': 'Marathi', 'mn': 'Mongolian', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'no': 'Norwegian', 'or': 'Odia',
    'ps': 'Pashto', 'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese', 'pa': 'Punjabi', 'ro': 'Romanian',
    'ru': 'Russian', 'sm': 'Samoan', 'gd': 'Scots Gaelic', 'sr': 'Serbian', 'st': 'Sesotho', 'sn': 'Shona',
    'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali', 'es': 'Spanish',
    'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish', 'tg': 'Tajik', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur', 'uz': 'Uzbek',
    'vi': 'Vietnamese', 'cy': 'Welsh', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
}
LANGUAGES_REVERSE = {v: k for k, v in LANGUAGES.items()}

# --- UI Styling ---
st.set_page_config(
    page_title="Translatica AI",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# --- Header with SVG Logo ---
st.markdown('<div class="translation-container">', unsafe_allow_html=True)
with open("Logo-full.svg", "r") as f:
    svg_logo = f.read()
st.markdown(
    f'''
    <div class="main-header">
        <div style="height:60px; width:auto; display:flex; align-items:center;">{svg_logo}</div>
    </div>
    ''',
    unsafe_allow_html=True
)

# --- Language selection row and text areas ---
col1, col2 = st.columns(2)

with col1:
    source_languages = {"Detect Language": "auto"}
    source_languages.update({v: k for k, v in LANGUAGES.items()})
    source_lang_options = list(source_languages.keys())
    if 'source_lang' not in st.session_state:
        st.session_state['source_lang'] = source_lang_options[0]
    source_lang_html = f'''<div class="dropdown"><form id="source_lang_form" autocomplete="off">
    <select id="source_lang_select" name="source_lang_select">
    {''.join([f'<option value="{opt}"'+(' selected' if st.session_state['source_lang']==opt else '')+f'>{opt}</option>' for opt in source_lang_options])}
    </select>
    <input type="submit" style="display:none;"/>
    </form></div>'''
    st.markdown(source_lang_html, unsafe_allow_html=True)
    source_lang_js = '''<script>
    const form = window.parent.document.querySelector('#source_lang_form');
    if(form){
      form.onsubmit = function(e){
        e.preventDefault();
        const val = form.querySelector('select').value;
        window.parent.postMessage({type:'streamlit:setComponentValue', key:'source_lang', value:val}, '*');
      }
      form.querySelector('select').onchange = function(){form.requestSubmit();}
    }
    </script>'''
    st.markdown(source_lang_js, unsafe_allow_html=True)
    input_text = st.text_area(
        "Input Text",
        height=400,
        placeholder="Type or paste your text here...",
        label_visibility="collapsed",
        key="input_text_area"
    )
    # Calculate words and characters
    words = len(input_text.split()) if input_text.strip() else 0
    chars = len(input_text)
    st.markdown(f"<div style='margin-top:4px; margin-bottom:8px; text-align:left; color:#444; font-size:15px;'>{words} words, {chars} characters</div>", unsafe_allow_html=True)

with col2:
    target_languages = {v: k for k, v in LANGUAGES.items()}
    target_lang_options = list(target_languages.keys())
    default_target = "English"
    if 'target_lang' not in st.session_state:
        st.session_state['target_lang'] = default_target
    target_lang_html = f'''<div class="dropdown"><form id="target_lang_form" autocomplete="off">
    <select id="target_lang_select" name="target_lang_select">
    {''.join([f'<option value="{opt}"'+(' selected' if st.session_state['target_lang']==opt else '')+f'>{opt}</option>' for opt in target_lang_options])}
    </select>
    <input type="submit" style="display:none;"/>
    </form></div>'''
    st.markdown(target_lang_html, unsafe_allow_html=True)
    target_lang_js = '''<script>
    const form = window.parent.document.querySelector('#target_lang_form');
    if(form){
      form.onsubmit = function(e){
        e.preventDefault();
        const val = form.querySelector('select').value;
        window.parent.postMessage({type:'streamlit:setComponentValue', key:'target_lang', value:val}, '*');
      }
      form.querySelector('select').onchange = function(){form.requestSubmit();}
    }
    </script>'''
    st.markdown(target_lang_js, unsafe_allow_html=True)
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    st.text_area(
        "Output Text",
        value=st.session_state.translated_text,
        height=400,
        disabled=True,
        label_visibility="collapsed",
        key="output_text"
    )

# --- Translate button ---
st.markdown("<div style='text-align: center; margin: 24px 0 2rem 0;'>", unsafe_allow_html=True)
if st.button("Translate", type="primary", use_container_width=False):
    if input_text.strip():
        try:
            with st.spinner("Translating..."):
                # Get language codes
                source_code = source_languages[st.session_state['source_lang']]
                target_code = target_languages[st.session_state['target_lang']]

                # Compose prompt for DeepSeek
                if source_code == "auto":
                    prompt = (
                        f"Detect the language of the following text and translate it to {target_code}. "
                        "Only provide the translation, no explanations or additional text:\n\n"
                        f"{input_text}\n\nTranslation:"
                    )
                else:
                    prompt = (
                        f"Translate the following text from {source_code} to {target_code}. "
                        "Only provide the translation, no explanations or additional text:\n\n"
                        f"{input_text}\n\nTranslation:"
                    )

                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024
                }
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                translation = result["choices"][0]["message"]["content"].strip()
                st.session_state.translated_text = translation
                st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
    else:
        st.warning("Please enter some text to translate.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Additional features ---
# (Removed metrics row for words and characters)

# --- Footer ---
st.markdown("""
<div style='text-align:center; color:#888; font-size:14px; margin-top:32px;'>
Transtaor AI doesn't store your data but uses AI models to process them.
</div>
""", unsafe_allow_html=True)