import SwiftUI
import Foundation
import UserNotifications

// Translation Service - moved inline to fix compilation
struct TranslationRequest: Codable {
    let text: String
    let source_lang: String
    let target_lang: String
    let translation_id: String?
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
    // For local testing: Use your computer's IP address
    // For production: Use "https://cursor-first-test.onrender.com"
    private let baseURL = "https://cursor-first-test.onrender.com" // Production deployment
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private var currentTask: Task<Void, Never>?
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    private var currentTranslationID: String?
    
    // UserDefaults keys for persistence
    private let textProcessingKey = "TextTranslationProcessing"
    private let textTranslationIDKey = "TextTranslationID"
    private let originalTextKey = "TextTranslationOriginalText"
    private let sourceLangKey = "TextTranslationSourceLang"
    private let targetLangKey = "TextTranslationTargetLang"
    
    private init() {
        // Request notification permissions
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
        
        // Check for ongoing translations on app launch
        checkForOngoingTranslations()
    }
    
    func translate(text: String, fromLanguage: String = "auto", toLanguage: String = "en") async -> Result<String, Error> {
        // Cancel any existing request
        currentTask?.cancel()
        
        // Generate unique translation ID
        let translationID = UUID().uuidString
        currentTranslationID = translationID
        
        // Save translation state for persistence
        saveTranslationState(
            isProcessing: true,
            translationID: translationID,
            originalText: text,
            sourceLang: fromLanguage,
            targetLang: toLanguage
        )
        
        isLoading = true
        errorMessage = nil
        
        // Begin background task
        beginBackgroundTask()
        
        defer {
            // End background task
            endBackgroundTask()
            
            // Clear persistence if not backgrounded
            if !isLoading {
                clearTranslationState()
            }
        }
        
        guard let url = URL(string: "\(baseURL)/translate") else {
            let error = NSError(domain: "TranslationService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            return .failure(error)
        }
        
        let requestBody = TranslationRequest(
            text: text,
            source_lang: fromLanguage,
            target_lang: toLanguage,
            translation_id: translationID
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
                    // Reset processing state on HTTP error
                    isLoading = false
                    clearTranslationState()
                    endBackgroundTask()
                    
                    let error = NSError(domain: "TranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    errorMessage = "Translation failed. Please try again."
                    return .failure(error)
                }
            }
            
            let translationResponse = try JSONDecoder().decode(TranslationResponse.self, from: data)
            
            // Reset processing state
            isLoading = false
            clearTranslationState()
            endBackgroundTask()
            
            // Send completion notification
            sendCompletionNotification(originalText: text, translatedText: translationResponse.translated_text, success: true)
            
            return .success(translationResponse.translated_text)
            
        } catch {
            if error.localizedDescription.contains("cancelled") {
                // Request was cancelled, don't show error
                return .failure(error)
            }
            
            // Reset processing state on error
            isLoading = false
            clearTranslationState()
            endBackgroundTask()
            
            errorMessage = "Network error. Please check your connection."
            
            // Send error notification
            sendCompletionNotification(originalText: text, translatedText: nil, success: false, error: error.localizedDescription)
            
            return .failure(error)
        }
    }
    
    func cancelCurrentRequest() {
        currentTask?.cancel()
        isLoading = false
        clearTranslationState()
        endBackgroundTask()
    }
    
    func cancelTranslation() {
        // Cancel the current request locally
        cancelCurrentRequest()
        
        // Also cancel on the backend if we have a translation ID
        if let translationID = currentTranslationID {
            Task {
                await cancelTranslationOnBackend(translationID: translationID)
            }
        }
    }
    
    private func cancelTranslationOnBackend(translationID: String) async {
        guard let url = URL(string: "\(baseURL)/cancel-translation") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        
        let body = "translation_id=\(translationID)"
        request.httpBody = body.data(using: .utf8)
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse {
                print("Backend cancellation response: \(httpResponse.statusCode)")
            }
        } catch {
            print("Failed to cancel translation on backend: \(error)")
        }
    }
    
    // MARK: - Background Task Management
    
    private func beginBackgroundTask() {
        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "TextTranslation") { [weak self] in
            // Background task is about to expire
            self?.endBackgroundTask()
        }
    }
    
    private func endBackgroundTask() {
        if backgroundTaskID != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskID)
            backgroundTaskID = .invalid
        }
    }
    
    // MARK: - State Persistence
    
    private func saveTranslationState(
        isProcessing: Bool,
        translationID: String,
        originalText: String,
        sourceLang: String,
        targetLang: String
    ) {
        UserDefaults.standard.set(isProcessing, forKey: textProcessingKey)
        UserDefaults.standard.set(translationID, forKey: textTranslationIDKey)
        UserDefaults.standard.set(originalText, forKey: originalTextKey)
        UserDefaults.standard.set(sourceLang, forKey: sourceLangKey)
        UserDefaults.standard.set(targetLang, forKey: targetLangKey)
    }
    
    private func clearTranslationState() {
        UserDefaults.standard.removeObject(forKey: textProcessingKey)
        UserDefaults.standard.removeObject(forKey: textTranslationIDKey)
        UserDefaults.standard.removeObject(forKey: originalTextKey)
        UserDefaults.standard.removeObject(forKey: sourceLangKey)
        UserDefaults.standard.removeObject(forKey: targetLangKey)
    }
    
    private func checkForOngoingTranslations() {
        let wasProcessing = UserDefaults.standard.bool(forKey: textProcessingKey)
        if wasProcessing {
            let savedTranslationID = UserDefaults.standard.string(forKey: textTranslationIDKey)
            
            // Restore state
            isLoading = true
            currentTranslationID = savedTranslationID
            
            // Show notification that translation is still in progress
            sendInProgressNotification()
        }
    }
    
    // MARK: - Notifications
    
    private func sendCompletionNotification(originalText: String, translatedText: String?, success: Bool, error: String? = nil) {
        let content = UNMutableNotificationContent()
        content.title = success ? "Translation Complete" : "Translation Failed"
        
        if success, let translated = translatedText {
            let preview = String(translated.prefix(50))
            content.body = "Text translated successfully: \"\(preview)\(translated.count > 50 ? "..." : "")\""
        } else {
            content.body = "Failed to translate text. \(error ?? "Please try again.")"
        }
        
        content.sound = success ? .default : .default
        
        let request = UNNotificationRequest(
            identifier: "text-translation-\(currentTranslationID ?? UUID().uuidString)",
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Notification error: \(error)")
            }
        }
    }
    
    private func sendInProgressNotification() {
        let content = UNMutableNotificationContent()
        content.title = "Translation in Progress"
        content.body = "Your text translation is still running in the background."
        content.sound = nil
        
        let request = UNNotificationRequest(
            identifier: "text-translation-in-progress",
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Notification error: \(error)")
            }
        }
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
        Language(code: "nl", name: "Dutch"),
        Language(code: "fr", name: "French"),
        Language(code: "de", name: "German"),
        Language(code: "hi", name: "Hindi"),
        Language(code: "it", name: "Italian"),
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
    
    // Consistent blue color to match buttons
    private let appBlue = Color.blue
    
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
            "nl": "nl",
            "fr": "fr",
            "de": "de",
            "hi": "hi",
            "it": "it",
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
                            .frame(height: 45) // 90 / 2 = 45
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
                                    .font(.caption)
                                    .foregroundColor(.primary)
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.secondary)
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
                            .foregroundColor(.secondary)
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
                                    .font(.caption)
                                    .foregroundColor(.primary)
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.secondary)
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
                                .padding(.horizontal, 0)
                                .padding(.vertical, 12)
                                .background(Color.clear)
                                .cornerRadius(12)
                                .onTapGesture {
                                    isFocused = true
                                }
                            
                            if inputText.isEmpty && !isFocused {
                                HStack {
                                    Text("Type or paste text here...")
                                        .font(.callout)
                                        .foregroundColor(.secondary)
                                        .padding(.horizontal, 0)
                                        .padding(.vertical, 20)
                                        .allowsHitTesting(false)
                                    Spacer()
                                }
                            }
                            
                            // Clear button - positioned in top-right corner
                            if !inputText.isEmpty {
                                VStack {
                                    HStack {
                                        Spacer()
                                        Button(action: {
                                            inputText = ""
                                            outputText = ""
                                            isTranslating = false
                                            debounceTimer?.invalidate()
                                            translationService.cancelCurrentRequest()
                                        }) {
                                            Image(systemName: "xmark.circle.fill")
                                                .foregroundColor(.secondary)
                                                .font(.system(size: 20))
                                        }
                                        .padding(.trailing, 8)
                                        .padding(.top, 8)
                                    }
                                    Spacer()
                                }
                            }
                        }
                    }
                    
                    // Output text area
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            if !outputText.isEmpty && !isTranslating {
                                Text(TranslationService.supportedLanguages.first(where: { $0.code == targetLang })?.name ?? "English")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            if !outputText.isEmpty && !isTranslating {
                                Button(action: copyTranslation) {
                                    HStack(spacing: 4) {
                                        Image(systemName: showCopyConfirmation ? "checkmark" : "doc.on.doc")
                                        Text(showCopyConfirmation ? "Copied!" : "Copy")
                                    }
                                    .font(.caption)
                                    .foregroundColor(showCopyConfirmation ? .green : appBlue)
                                }
                            }
                        }
                        
                        ZStack {
                            ScrollView {
                                VStack {
                                    if isTranslating {
                                        VStack(spacing: 16) {
                                            ProgressView()
                                                .scaleEffect(1.2)
                                            Text("Translating...")
                                                .font(.body)
                                                .foregroundColor(.secondary)
                                            
                                            // Cancel button
                                            Button(action: {
                                                translationService.cancelTranslation()
                                                isTranslating = false
                                            }) {
                                                HStack {
                                                    Image(systemName: "xmark.circle.fill")
                                                    Text("Cancel")
                                                }
                                                .font(.caption)
                                                .foregroundColor(.white)
                                                .padding(.horizontal, 16)
                                                .padding(.vertical, 8)
                                                .background(Color.red)
                                                .cornerRadius(8)
                                            }
                                        }
                                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                                        .padding()
                                    } else if !outputText.isEmpty {
                                        Text(outputText)
                                            .font(.body)
                                            .fontWeight(.medium)
                                            .foregroundColor(.primary)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .padding(.horizontal, 0)
                                            .padding(.vertical, 12)
                                    }
                                }
                            }
                        }
                        .frame(minHeight: 200)
                        .cornerRadius(12)
                    }
                }
                .padding()
            }
            .navigationBarHidden(true)
            .onAppear {
                // Load saved language preferences on app launch
                loadSavedLanguages()
                // Ensure text field is not focused on launch
                isFocused = false
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
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

