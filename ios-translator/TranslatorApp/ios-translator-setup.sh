#!/bin/bash

echo "🚀 iOS Translator Setup Script"
echo "==============================="
echo ""
echo "This script will set up your translation files after you create the basic Xcode project."
echo ""
echo "STEP 1: First create a new iOS project in Xcode:"
echo "  1. Open Xcode"
echo "  2. File → New → Project"
echo "  3. iOS → App"
echo "  4. Product Name: TranslatorApp"
echo "  5. Bundle ID: com.YOURNAME.TranslatorApp"
echo "  6. Language: Swift, Interface: SwiftUI"
echo "  7. Save to: /Users/yogesh/Cursor_first_test/ios-translator"
echo ""
echo "STEP 2: Add Share Extension:"
echo "  1. File → New → Target"
echo "  2. iOS → Share Extension"
echo "  3. Product Name: TranslateExtension"
echo "  4. Language: Swift"
echo ""
echo "STEP 3: After creating the project, run this script again with 'setup' parameter:"
echo "  ./ios-translator-setup.sh setup"
echo ""

if [ "$1" != "setup" ]; then
    echo "Please complete steps 1-2 above, then run:"
    echo "./ios-translator-setup.sh setup"
    exit 0
fi

echo "🔧 Setting up translation files..."

# Check if we're in the right directory
if [ ! -d "TranslatorApp.xcodeproj" ]; then
    echo "❌ Please run this script from your ios-translator directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "✅ Found Xcode project"

# Create the shared translation service
echo "📝 Creating TranslationService.swift..."
mkdir -p Shared

cat > Shared/TranslationService.swift << 'EOF'
import Foundation

struct TranslationRequest: Codable {
    let text: String
    let source_lang: String
    let target_lang: String
}

struct TranslationResponse: Codable {
    let translation: String
}

class TranslationService: ObservableObject {
    static let shared = TranslationService()
    
    // Update this URL to match your deployment
    private let baseURL = "https://cursor-first-test.onrender.com"
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private init() {}
    
    func translate(text: String, fromLanguage: String = "auto", toLanguage: String = "en") async -> Result<String, Error> {
        isLoading = true
        errorMessage = nil
        
        defer {
            DispatchQueue.main.async {
                self.isLoading = false
            }
        }
        
        guard let url = URL(string: "\(baseURL)/translate") else {
            let error = NSError(domain: "TranslationService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            return .failure(error)
        }
        
        let requestBody = TranslationRequest(
            text: text,
            source_lang: fromLanguage,
            target_lang: toLanguage
        )
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "TranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    DispatchQueue.main.async {
                        self.errorMessage = "Translation failed. Please try again."
                    }
                    return .failure(error)
                }
            }
            
            let translationResponse = try JSONDecoder().decode(TranslationResponse.self, from: data)
            return .success(translationResponse.translation)
            
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Network error. Please check your connection."
            }
            return .failure(error)
        }
    }
}

// Language definitions matching your web app
struct Language {
    let code: String
    let name: String
}

extension TranslationService {
    static let supportedLanguages = [
        Language(code: "en", name: "English"),
        Language(code: "ar", name: "Arabic"),  
        Language(code: "zh-cn", name: "Chinese (Simplified)"),
        Language(code: "zh-tw", name: "Chinese (Traditional)"),
        Language(code: "nl", name: "Dutch"),
        Language(code: "fr", name: "French"),
        Language(code: "de", name: "German"),
        Language(code: "hi", name: "Hindi"),
        Language(code: "it", name: "Italian"),
        Language(code: "ja", name: "Japanese"),
        Language(code: "ko", name: "Korean"),
        Language(code: "pl", name: "Polish"),
        Language(code: "pt", name: "Portuguese"),
        Language(code: "ru", name: "Russian"),
        Language(code: "es", name: "Spanish"),
        Language(code: "tr", name: "Turkish"),
        Language(code: "uk", name: "Ukrainian"),
        Language(code: "vi", name: "Vietnamese")
    ]
}
EOF

# Update ContentView.swift
echo "📝 Updating ContentView.swift..."
cat > TranslatorApp/ContentView.swift << 'EOF'
import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                Image(systemName: "globe")
                    .font(.system(size: 64))
                    .foregroundColor(.blue)
                
                Text("AI Translator")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                VStack(spacing: 16) {
                    Text("Translate text from any app!")
                        .font(.title2)
                        .multilineTextAlignment(.center)
                    
                    Text("To use the translator:")
                        .font(.headline)
                        .padding(.top)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "1.circle.fill")
                                .foregroundColor(.blue)
                            Text("Select text in any app")
                        }
                        
                        HStack {
                            Image(systemName: "2.circle.fill")
                                .foregroundColor(.blue)
                            Text("Tap \"Share\" or \"Copy\"")
                        }
                        
                        HStack {
                            Image(systemName: "3.circle.fill")
                                .foregroundColor(.blue)
                            Text("Choose \"Translate with AI\"")
                        }
                        
                        HStack {
                            Image(systemName: "4.circle.fill")
                                .foregroundColor(.blue)
                            Text("Select target language and translate!")
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(12)
                }
                
                Spacer()
                
                Text("Made with ❤️ using AI")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .navigationTitle("AI Translator")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
EOF

echo "✅ Setup complete!"
echo ""
echo "🎯 Next Steps:"
echo "1. In Xcode, add the Shared/TranslationService.swift file to both targets:"
echo "   - Right-click project → Add Files to \"TranslatorApp\""
echo "   - Select Shared/TranslationService.swift"
echo "   - Make sure BOTH targets are checked"
echo ""
echo "2. Replace the ShareViewController.swift content (will provide separately)"
echo ""
echo "3. Build and run! (⌘+R)"

chmod +x ios-translator-setup.sh 