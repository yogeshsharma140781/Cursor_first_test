import SwiftUI
import Foundation

// Translation Service - moved inline to fix compilation
struct TranslationRequest: Codable {
    let text: String
    let source_lang: String
    let target_lang: String
}

struct TranslationResponse: Codable {
    let translated_text: String
    let original_text: String
    let source_lang: String
    let target_lang: String
    let status: String
}

@MainActor
class TranslationService: ObservableObject {
    static let shared = TranslationService()
    
    // Update this URL to match your deployment
    // For local testing: Use your computer's IP address
    // For production: Use "https://cursor-first-test.onrender.com"
    private let baseURL = "https://cursor-first-test.onrender.com" // Production deployment
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private var currentTask: Task<Void, Never>?
    
    private init() {}
    
    func translate(text: String, fromLanguage: String = "auto", toLanguage: String = "en") async -> Result<String, Error> {
        // Cancel any existing request
        currentTask?.cancel()
        
        isLoading = true
        errorMessage = nil
        
        defer {
            isLoading = false
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
        request.timeoutInterval = 30.0 // 30 second timeout
        
        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "TranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    errorMessage = "Translation failed. Please try again."
                    return .failure(error)
                }
            }
            
            let translationResponse = try JSONDecoder().decode(TranslationResponse.self, from: data)
            return .success(translationResponse.translated_text)
            
        } catch {
            if error.localizedDescription.contains("cancelled") {
                // Request was cancelled, don't show error
                return .failure(error)
            }
            errorMessage = "Network error. Please check your connection."
            return .failure(error)
        }
    }
    
    func cancelCurrentRequest() {
        currentTask?.cancel()
        isLoading = false
    }
}

// Language definitions matching your web app
struct Language: Sendable {
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

struct ContentView: View {
    @StateObject private var translationService = TranslationService.shared
    @State private var sourceLang = "auto"
    @State private var targetLang = ""
    @State private var inputText = ""
    @State private var outputText = ""
    @State private var isTranslating = false
    @State private var isFocused = false
    @State private var debounceTimer: Timer?
    @State private var showCopyConfirmation = false
    
    // UserDefaults keys for saving language preferences
    private let sourceLanguageKey = "LastUsedSourceLanguage"
    private let targetLanguageKey = "LastUsedTargetLanguage"
    
    // Computed properties for word and character count
    private var wordCount: Int {
        inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0 : inputText.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: .whitespacesAndNewlines).count
    }
    
    private var charCount: Int {
        inputText.count
    }
    
    // Function to get device language and map to supported language
    private func getDeviceLanguage() -> String {
        let deviceLanguage = Locale.current.languageCode ?? "en"
        
        // Map device language codes to our supported language codes
        let languageMapping: [String: String] = [
            "en": "en",
            "ar": "ar",
            "zh": "zh-cn",
            "nl": "nl",
            "fr": "fr",
            "de": "de",
            "hi": "hi",
            "it": "it",
            "ja": "ja",
            "ko": "ko",
            "pl": "pl",
            "pt": "pt",
            "ru": "ru",
            "es": "es",
            "tr": "tr",
            "uk": "uk",
            "vi": "vi"
        ]
        
        return languageMapping[deviceLanguage] ?? "en"
    }
    
    // Function to load saved language preferences
    private func loadSavedLanguages() {
        let savedSourceLang = UserDefaults.standard.string(forKey: sourceLanguageKey)
        let savedTargetLang = UserDefaults.standard.string(forKey: targetLanguageKey)
        
        // Use saved source language or default to "auto"
        sourceLang = savedSourceLang ?? "auto"
        
        // Use saved target language, or device language, or fallback to English
        if let saved = savedTargetLang {
            targetLang = saved
        } else {
            targetLang = getDeviceLanguage()
        }
    }
    
    // Function to save language preferences
    private func saveLanguagePreferences() {
        UserDefaults.standard.set(sourceLang, forKey: sourceLanguageKey)
        UserDefaults.standard.set(targetLang, forKey: targetLanguageKey)
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header with app logo
                    VStack(spacing: 8) {
                        Image("Logo-full")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(height: 50)
                    }
                    .padding(.top)
                    
                    // Language selectors - full width
                    HStack(spacing: 12) {
                        // Source Language
                        Menu {
                            Button("Detect") {
                                sourceLang = "auto"
                            }
                            ForEach(TranslationService.supportedLanguages, id: \.code) { language in
                                Button(language.name) {
                                    sourceLang = language.code
                                }
                            }
                        } label: {
                            HStack {
                                Text(sourceLang == "auto" ? "Detect" : (TranslationService.supportedLanguages.first(where: { $0.code == sourceLang })?.name ?? "Detect"))
                                    .foregroundColor(.blue)
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.blue)
                                    .font(.caption)
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 12)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                        }
                        .frame(maxWidth: .infinity)
                        
                        // Arrow
                        Image(systemName: "arrow.right")
                            .foregroundColor(.blue)
                            .font(.caption)
                        
                        // Target Language
                        Menu {
                            ForEach(TranslationService.supportedLanguages, id: \.code) { language in
                                Button(language.name) {
                                    targetLang = language.code
                                }
                            }
                        } label: {
                            HStack {
                                Text(TranslationService.supportedLanguages.first(where: { $0.code == targetLang })?.name ?? "English")
                                    .foregroundColor(.blue)
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.blue)
                                    .font(.caption)
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 12)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    
                    // Input text area
                    VStack(alignment: .leading, spacing: 8) {
                        ZStack(alignment: .topLeading) {
                            TextEditor(text: $inputText)
                                .frame(minHeight: 120)
                                .padding(12)
                                .background(Color.clear)
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(isFocused ? Color.blue : Color.gray.opacity(0.3), lineWidth: 1)
                                )
                                .onTapGesture {
                                    isFocused = true
                                }
                            
                            if inputText.isEmpty && !isFocused {
                                Text("Type or paste text here...")
                                    .foregroundColor(.gray)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 20)
                                    .allowsHitTesting(false)
                            }
                        }
                        
                        // Word and character count
                        HStack {
                            Text("\(wordCount) words, \(charCount) characters")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                        }
                    }
                    
                    // Output text area
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Translation")
                                .font(.headline)
                                .foregroundColor(.primary)
                            
                            Spacer()
                            
                            if !outputText.isEmpty && !isTranslating {
                                Button(action: copyTranslation) {
                                    HStack(spacing: 4) {
                                        Image(systemName: showCopyConfirmation ? "checkmark" : "doc.on.doc")
                                        Text(showCopyConfirmation ? "Copied!" : "Copy")
                                    }
                                    .font(.caption)
                                    .foregroundColor(showCopyConfirmation ? .green : .blue)
                                }
                            }
                        }
                        
                        ZStack {
                            ScrollView {
                                VStack {
                                    if isTranslating {
                                        VStack(spacing: 12) {
                                            ProgressView()
                                                .scaleEffect(1.2)
                                            Text("Translating...")
                                                .font(.body)
                                                .foregroundColor(.secondary)
                                        }
                                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                                        .padding()
                                    } else if !outputText.isEmpty {
                                        Text(outputText)
                                            .font(.body)
                                            .foregroundColor(.blue)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .padding()
                                    }
                                }
                            }
                        }
                        .frame(minHeight: 200)
                        .background(Color.gray.opacity(0.05))
                        .cornerRadius(12)
                    }
                }
                .padding()
            }
            .navigationBarHidden(true)
            .onAppear {
                // Load saved language preferences on app launch
                loadSavedLanguages()
            }
            .onChange(of: inputText) { _ in
                performDebouncedTranslation()
            }
            .onChange(of: sourceLang) { _ in
                // Save language preference when changed
                saveLanguagePreferences()
                if !inputText.isEmpty {
                    performDebouncedTranslation()
                }
            }
            .onChange(of: targetLang) { _ in
                // Save language preference when changed
                saveLanguagePreferences()
                if !inputText.isEmpty {
                    performDebouncedTranslation()
                }
            }
            .onTapGesture {
                // Dismiss keyboard and unfocus
                isFocused = false
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
    }
    
    private func performDebouncedTranslation() {
        // Cancel previous timer and any ongoing translation
        debounceTimer?.invalidate()
        translationService.cancelCurrentRequest()
        
        // Clear output if input is empty
        if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            outputText = ""
            isTranslating = false
            return
        }
        
        // Set up new timer with 1.2 second delay (matching web app)
        debounceTimer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: false) { _ in
            Task {
                await performTranslation()
            }
        }
    }
    
    private func performTranslation() async {
        let textToTranslate = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !textToTranslate.isEmpty else { return }
        
        await MainActor.run {
            isTranslating = true
            outputText = ""
        }
        
        let result = await translationService.translate(
            text: textToTranslate,
            fromLanguage: sourceLang,
            toLanguage: targetLang
        )
        
        await MainActor.run {
            isTranslating = false
            
            switch result {
            case .success(let translation):
                outputText = translation
            case .failure(let error):
                if !error.localizedDescription.contains("cancelled") {
                    outputText = "Translation failed. Please try again."
                    print("Translation error: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func copyTranslation() {
        UIPasteboard.general.string = outputText
        
        // Show confirmation
        showCopyConfirmation = true
        
        // Hide confirmation after 2 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            showCopyConfirmation = false
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

