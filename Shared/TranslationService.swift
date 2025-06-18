import Foundation

struct TranslationRequest: Codable {
    let text: String
    let source_lang: String
    let target_lang: String
}

struct TranslationResponse: Codable {
    let translation: String
}

@MainActor
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
            Task { @MainActor in
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
                    await MainActor.run {
                        self.errorMessage = "Translation failed. Please try again."
                    }
                    return .failure(error)
                }
            }
            
            let translationResponse = try JSONDecoder().decode(TranslationResponse.self, from: data)
            return .success(translationResponse.translation)
            
        } catch {
            await MainActor.run {
                self.errorMessage = "Network error. Please check your connection."
            }
            return .failure(error)
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