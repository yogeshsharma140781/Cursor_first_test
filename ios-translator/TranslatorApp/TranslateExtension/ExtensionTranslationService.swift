import Foundation

// Translation Service for Share Extension
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
            return .success(translationResponse.translation)
            
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

// Language definitions for Share Extension
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

// PDF Translation Service for Share Extension
@MainActor
class PDFTranslationService: ObservableObject {
    static let shared = PDFTranslationService()
    
    // Update this URL to match your deployment
    private let baseURL = "https://cursor-first-test.onrender.com"
    
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var progress: Double = 0.0
    
    private var currentTask: Task<Void, Never>?
    
    private init() {}
    
    func translatePDF(
        pdfData: Data,
        filename: String,
        fromLanguage: String = "auto",
        toLanguage: String = "en"
    ) async -> Result<Data, Error> {
        // Cancel any existing request
        currentTask?.cancel()
        
        isProcessing = true
        errorMessage = nil
        progress = 0.0
        
        defer {
            isProcessing = false
            progress = 0.0
        }
        
        guard let url = URL(string: "\(baseURL)/translate-pdf") else {
            let error = NSError(domain: "PDFTranslationService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            return .failure(error)
        }
        
        // Create multipart form data request
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300.0 // 5 minute timeout for PDF processing
        
        // Build multipart body
        var body = Data()
        
        // Add source language parameter
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"source_lang\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(fromLanguage)\r\n".data(using: .utf8)!)
        
        // Add target language parameter
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"target_lang\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(toLanguage)\r\n".data(using: .utf8)!)
        
        // Add PDF file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: application/pdf\r\n\r\n".data(using: .utf8)!)
        body.append(pdfData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Close boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        do {
            progress = 0.3 // Upload complete
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            progress = 0.8 // Processing complete
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "PDFTranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    errorMessage = "PDF translation failed. Please try again."
                    return .failure(error)
                }
            }
            
            progress = 1.0 // Complete
            return .success(data)
            
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
        isProcessing = false
        progress = 0.0
    }
} 