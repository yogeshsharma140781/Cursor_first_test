#!/bin/bash

# Setup script for Unicode font support in PDF translation system
# This script installs necessary fonts and dependencies

echo "Setting up Unicode Font Support for PDF Translation"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_status "Detected OS: $MACHINE"

# Install system dependencies
install_system_fonts() {
    print_status "Installing system fonts..."
    
    case $MACHINE in
        Linux)
            # Try different package managers
            if command -v apt-get &> /dev/null; then
                print_status "Using apt-get to install fonts..."
                sudo apt-get update
                sudo apt-get install -y fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-liberation
            elif command -v yum &> /dev/null; then
                print_status "Using yum to install fonts..."
                sudo yum install -y google-noto-fonts-common google-noto-sans-fonts google-noto-cjk-fonts
            elif command -v pacman &> /dev/null; then
                print_status "Using pacman to install fonts..."
                sudo pacman -S --noconfirm noto-fonts noto-fonts-cjk noto-fonts-emoji
            else
                print_warning "No supported package manager found. Will download fonts manually."
                return 1
            fi
            ;;
        Mac)
            print_status "macOS detected. Checking for Homebrew..."
            if command -v brew &> /dev/null; then
                print_status "Installing fonts via Homebrew..."
                brew tap homebrew/cask-fonts
                brew install --cask font-noto-sans font-noto-sans-cjk
            else
                print_warning "Homebrew not found. Will download fonts manually."
                return 1
            fi
            ;;
        *)
            print_warning "Unsupported OS for automatic font installation. Will download fonts manually."
            return 1
            ;;
    esac
    
    return 0
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found. Make sure you're in the backend directory."
        return 1
    fi
}

# Download fonts manually if system installation fails
download_fonts_manually() {
    print_status "Downloading fonts manually..."
    python3 install_fonts.py
}

# Test the installation
test_installation() {
    print_status "Testing Unicode font support..."
    python3 test_unicode_fonts.py
    
    if [ $? -eq 0 ]; then
        print_status "Test completed successfully!"
        print_status "Check the generated 'unicode_font_test.pdf' file to verify font rendering."
    else
        print_error "Test failed. Check the error messages above."
        return 1
    fi
}

# Main installation process
main() {
    print_status "Starting Unicode font setup..."
    
    # Check if we're in the right directory
    if [ ! -f "translator_api.py" ]; then
        print_error "translator_api.py not found. Please run this script from the backend directory."
        exit 1
    fi
    
    # Install Python dependencies
    if ! install_python_deps; then
        print_error "Failed to install Python dependencies."
        exit 1
    fi
    
    # Try system font installation first
    if ! install_system_fonts; then
        print_warning "System font installation failed or not available."
        print_status "Falling back to manual font download..."
        download_fonts_manually
    fi
    
    # Test the installation
    if test_installation; then
        print_status ""
        print_status "✅ Unicode font support setup completed successfully!"
        print_status ""
        print_status "Your PDF translation system now supports:"
        print_status "• Latin scripts (English, European languages)"
        print_status "• Chinese, Japanese, Korean (CJK)"
        print_status "• Hindi and other Devanagari scripts"  
        print_status "• Arabic script"
        print_status "• Cyrillic script (Russian, etc.)"
        print_status ""
        print_status "Restart your translation API server to use the new fonts."
    else
        print_error "Setup completed with errors. Check the logs above."
        exit 1
    fi
}

# Run main function
main 