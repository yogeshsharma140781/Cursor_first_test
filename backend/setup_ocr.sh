#!/bin/bash

# Setup script for OCR functionality
# This script installs Tesseract OCR and language packs

echo "🔧 Setting up OCR functionality..."

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "📱 Detected OS: $OS"

# Install Tesseract OCR based on OS
case $OS in
    "linux")
        echo "🐧 Installing Tesseract for Linux..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
        
        # Install language packs
        echo "🌍 Installing language packs..."
        sudo apt-get install -y \
            tesseract-ocr-eng \
            tesseract-ocr-fra \
            tesseract-ocr-deu \
            tesseract-ocr-spa \
            tesseract-ocr-ita \
            tesseract-ocr-por \
            tesseract-ocr-rus \
            tesseract-ocr-ara \
            tesseract-ocr-hin \
            tesseract-ocr-nld \
            tesseract-ocr-pol \
            tesseract-ocr-tur \
            tesseract-ocr-ukr \
            tesseract-ocr-vie
        ;;
        
    "macos")
        echo "🍎 Installing Tesseract for macOS..."
        if command -v brew &> /dev/null; then
            brew install tesseract
            
            # Install language packs
            echo "🌍 Installing language packs..."
            brew install tesseract-lang
        else
            echo "❌ Homebrew not found. Please install Homebrew first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        ;;
        
    "windows")
        echo "🪟 For Windows, please install Tesseract manually:"
        echo "   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki"
        echo "   2. Install and add to PATH"
        echo "   3. Ensure language packs are included during installation"
        ;;
        
    *)
        echo "❌ Unsupported OS. Please install Tesseract manually."
        exit 1
        ;;
esac

# Install additional image processing tools
case $OS in
    "linux")
        echo "🖼️ Installing additional image processing tools..."
        sudo apt-get install -y \
            poppler-utils \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxrender1 \
            libxext6
        ;;
        
    "macos")
        echo "🖼️ Installing additional image processing tools..."
        brew install poppler
        ;;
esac

# Test Tesseract installation
echo "🧪 Testing Tesseract installation..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract installed successfully!"
    tesseract --version
    
    echo "🔍 Available languages:"
    tesseract --list-langs
else
    echo "❌ Tesseract installation failed or not in PATH"
    exit 1
fi

echo ""
echo "🎉 OCR setup completed successfully!"
echo "📝 You can now use the scanned PDF translation feature."
echo ""
echo "🚀 To start the server:"
echo "   cd backend"
echo "   pip install -r requirements.txt"
echo "   python translator_api.py" 