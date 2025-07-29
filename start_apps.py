#!/usr/bin/env python3
"""
Startup script for translation applications.
This script helps you choose and start the appropriate application.
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import httpx
        import dotenv
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_streamlit_app():
    """Start the Streamlit application."""
    print("🚀 Starting Streamlit Translation App...")
    print("📝 This app uses manual translation (click the Translate button)")
    print("🌐 Opening in your default browser...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "deepseek_translator_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ Streamlit app stopped.")

def start_react_backend():
    """Start the FastAPI backend for React app."""
    print("🚀 Starting FastAPI Backend...")
    print("🔧 Backend will run on http://localhost:8000")
    
    os.chdir("backend")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"])
    except KeyboardInterrupt:
        print("\n✅ Backend stopped.")

def start_react_frontend():
    """Start the React frontend."""
    print("🚀 Starting React Frontend...")
    print("🌐 Frontend will run on http://localhost:3000")
    
    os.chdir("frontend")
    try:
        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n✅ Frontend stopped.")

def main():
    print("🌟 Translation App Launcher")
    print("=" * 40)
    
    if not check_requirements():
        return
    
    print("\nChoose which application to start:")
    print("1. Streamlit App (Simple, single file)")
    print("2. FastAPI Backend (for React app)")  
    print("3. React Frontend (requires backend running)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            start_streamlit_app()
            break
        elif choice == "2":
            start_react_backend()
            break
        elif choice == "3":
            print("⚠️  Make sure the backend is running first!")
            input("Press Enter to continue...")
            start_react_frontend()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main() 