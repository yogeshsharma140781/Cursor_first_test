# 📱 iOS AI Translator Setup Guide

## ✅ **What This Creates**
A native iOS app with Share Extension that lets users translate text from any app using your existing translation API.

## 🚀 **Quick Setup (5 Steps)**

### **Step 1: Create New iOS Project in Xcode**
1. Open **Xcode**
2. Choose **"Create a new Xcode project"**
3. Select **iOS** → **App**
4. Fill in:
   - **Product Name:** `TranslatorApp`
   - **Bundle Identifier:** `com.YOURNAME.TranslatorApp` (replace YOURNAME with your name)
   - **Language:** Swift
   - **Interface:** SwiftUI
   - **Use Core Data:** Leave unchecked
   - **Include Tests:** Leave unchecked (optional)
5. **Save to:** `/Users/yogesh/Cursor_first_test/ios-translator/`
6. Click **"Create"**

### **Step 2: Add Share Extension Target**
1. In Xcode, go to **File** → **New** → **Target...**
2. Select **iOS** → **Share Extension**
3. Fill in:
   - **Product Name:** `TranslateExtension`
   - **Language:** Swift
4. Click **"Finish"**
5. When asked **"Activate TranslateExtension scheme?"**, click **"Activate"**

### **Step 3: Run the Setup Script**
1. Open Terminal and navigate to your project:
   ```bash
   cd /Users/yogesh/Cursor_first_test
   ./ios-translator-setup.sh setup
   ```

This will create all the necessary files including:
- `Shared/TranslationService.swift` - API communication
- Updated `TranslatorApp/ContentView.swift` - Main app UI

### **Step 4: Add Files to Xcode**
1. In Xcode, **right-click** on your project name in the navigator
2. Choose **"Add Files to TranslatorApp"**
3. Navigate to and select the `Shared` folder
4. **IMPORTANT:** In the dialog, make sure **BOTH** targets are checked:
   - ✅ TranslatorApp
   - ✅ TranslateExtension
5. Click **"Add"**

### **Step 5: Replace ShareViewController Content**
1. In Xcode, open `TranslateExtension/ShareViewController.swift`
2. **Delete all the existing content**
3. **Copy and paste** the entire content from the `ShareViewController-content.swift` file

## 🔧 **Configure Signing (Required)**

### **Set Development Team:**
1. Select **TranslatorApp** project in navigator
2. Select **TranslatorApp** target
3. Go to **"Signing & Capabilities"** tab
4. Set **"Team"** to your Apple Developer account

5. Select **TranslateExtension** target
6. Set the **same "Team"** as above

**⚠️ Important:** Both targets must use the same signing team!

## ▶️ **Build and Run**

1. Select a device/simulator from the dropdown (iPhone 15, etc.)
2. Press **⌘+R** to build and run
3. The app should install and open successfully

## 🧪 **Test the Share Extension**

1. Open **Safari** or **Notes** app
2. **Select some text**
3. Tap **"Share"** from the selection menu
4. Look for **"Translate with AI"** in the share sheet
5. Tap it to test translation!

## 🎯 **How It Works**

- **Main App:** Shows instructions and info about the translator
- **Share Extension:** Appears when text is selected in any app
- **Translation Service:** Connects to your API at `https://cursor-first-test.onrender.com`
- **18 Languages:** Supports all the same languages as your web app

## 🔧 **Troubleshooting**

### **"Share Extension doesn't appear"**
- Make sure both targets use the same signing team
- Clean build folder: **⌘+Shift+K**, then rebuild

### **"Build failed"**
- Check that `TranslationService.swift` is added to both targets
- Verify all files are in the correct locations

### **"Translation not working"**
- Check your API is running at `https://cursor-first-test.onrender.com`
- Check device/simulator has internet connection
- Look at Xcode console for error messages

## 📁 **File Structure**
```
ios-translator/
├── TranslatorApp.xcodeproj/
├── TranslatorApp/
│   ├── TranslatorAppApp.swift
│   ├── ContentView.swift (updated)
│   └── Assets.xcassets/
├── TranslateExtension/
│   ├── ShareViewController.swift (replace content)
│   ├── Info.plist
│   └── Base.lproj/MainInterface.storyboard
└── Shared/
    └── TranslationService.swift (add to both targets)
```

## 🚀 **After Setup**

Your iOS app will:
- ✅ Appear as "Translate with AI" in share menus
- ✅ Auto-translate selected text
- ✅ Support 18 languages
- ✅ Allow language selection
- ✅ Provide copy-to-clipboard functionality
- ✅ Work from any iOS app (Safari, Messages, Notes, etc.)

## 🎉 **Success!**

You now have a fully functional iOS translation app that integrates with your existing translation service!

---

**Need help?** Check the troubleshooting section above or verify all files are in the correct locations. 