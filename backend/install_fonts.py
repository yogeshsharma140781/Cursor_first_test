#!/usr/bin/env python3
"""
Script to download and install Unicode fonts for PDF translation support
This ensures proper rendering of non-Roman scripts like Hindi, Japanese, Arabic, etc.
"""

import os
import requests
import platform
import tempfile
import zipfile
import shutil
from pathlib import Path

def get_fonts_directory():
    """Get the appropriate fonts directory for the current OS"""
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        fonts_dir = Path.home() / 'Library' / 'Fonts'
    elif system == 'linux':
        fonts_dir = Path.home() / '.fonts'
    else:  # Windows
        fonts_dir = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'Windows' / 'Fonts'
    
    fonts_dir.mkdir(parents=True, exist_ok=True)
    return fonts_dir

def download_file(url, filename, fonts_dir):
    """Download a file to the fonts directory"""
    file_path = fonts_dir / filename
    
    if file_path.exists():
        print(f"Font already exists: {filename}")
        return True
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def download_google_fonts():
    """Download essential Unicode fonts from Google Fonts"""
    fonts_dir = get_fonts_directory()
    
    # List of essential Unicode fonts with their download URLs
    font_downloads = [
        # Noto Sans (Latin + extended)
        {
            'name': 'Noto Sans Regular',
            'filename': 'NotoSans-Regular.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf'
        },
        {
            'name': 'Noto Sans Bold',
            'filename': 'NotoSans-Bold.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf'
        },
        {
            'name': 'Noto Sans Italic',
            'filename': 'NotoSans-Italic.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Italic.ttf'
        },
        {
            'name': 'Noto Sans Bold Italic',
            'filename': 'NotoSans-BoldItalic.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-BoldItalic.ttf'
        },
        
        # Noto Sans CJK (Chinese, Japanese, Korean)
        {
            'name': 'Noto Sans CJK Regular',
            'filename': 'NotoSansCJK-Regular.ttc',
            'url': 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc'
        },
        {
            'name': 'Noto Sans CJK Bold',
            'filename': 'NotoSansCJK-Bold.ttc',
            'url': 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Bold.ttc'
        },
        
        # Noto Sans Devanagari (Hindi)
        {
            'name': 'Noto Sans Devanagari Regular',
            'filename': 'NotoSansDevanagari-Regular.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf'
        },
        {
            'name': 'Noto Sans Devanagari Bold',
            'filename': 'NotoSansDevanagari-Bold.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf'
        },
        
        # Noto Sans Arabic
        {
            'name': 'Noto Sans Arabic Regular',
            'filename': 'NotoSansArabic-Regular.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf'
        },
        {
            'name': 'Noto Sans Arabic Bold',
            'filename': 'NotoSansArabic-Bold.ttf',
            'url': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Bold.ttf'
        }
    ]
    
    print(f"Installing fonts to: {fonts_dir}")
    
    successful_downloads = 0
    for font in font_downloads:
        success = download_file(font['url'], font['filename'], fonts_dir)
        if success:
            successful_downloads += 1
    
    print(f"\nFont installation summary:")
    print(f"Successfully installed: {successful_downloads}/{len(font_downloads)} fonts")
    
    if successful_downloads > 0:
        print("\nFonts installed successfully! The PDF translator should now support:")
        print("- Latin scripts (English, European languages)")
        print("- Chinese, Japanese, Korean (CJK)")
        print("- Hindi and other Devanagari scripts")
        print("- Arabic script")
        print("\nRestart your translation service to use the new fonts.")
    else:
        print("\nWarning: No fonts were downloaded. Check your internet connection.")
        print("The service will fall back to system fonts.")

def install_system_fonts():
    """Install fonts using system package manager (Linux only)"""
    system = platform.system().lower()
    
    if system == 'linux':
        try:
            print("Attempting to install fonts via system package manager...")
            
            # Try different package managers
            install_commands = [
                'sudo apt-get update && sudo apt-get install -y fonts-noto fonts-noto-cjk fonts-noto-color-emoji',
                'sudo yum install -y google-noto-fonts-common google-noto-sans-fonts google-noto-cjk-fonts',
                'sudo pacman -S --noconfirm noto-fonts noto-fonts-cjk noto-fonts-emoji'
            ]
            
            for cmd in install_commands:
                result = os.system(cmd + ' 2>/dev/null')
                if result == 0:
                    print("Successfully installed system fonts!")
                    return True
            
            print("Could not install via package manager, using direct download...")
            return False
        except Exception as e:
            print(f"System font installation failed: {e}")
            return False
    else:
        print(f"System font installation not supported on {system}")
        return False

def main():
    """Main function to install Unicode fonts"""
    print("Unicode Font Installer for PDF Translation")
    print("=" * 50)
    
    # Try system package manager first (Linux only)
    if platform.system().lower() == 'linux':
        if install_system_fonts():
            print("\nSystem fonts installed successfully!")
            return
    
    # Fall back to direct download
    print("Installing fonts via direct download...")
    download_google_fonts()

if __name__ == "__main__":
    main() 