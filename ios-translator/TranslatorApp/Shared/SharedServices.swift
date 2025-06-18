import Foundation

// MARK: - Shared Translation Models
public struct TranslationRequest: Codable {
    public let text: String
    public let source_lang: String
    public let target_lang: String
    
    public init(text: String, source_lang: String, target_lang: String) {
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
    }
}

public struct TranslationResponse: Codable {
    public let translated_text: String
    public let original_text: String
    public let source_lang: String
    public let target_lang: String
    public let status: String
}

// MARK: - Language Support
public struct Language: Sendable {
    public let code: String
    public let name: String
    
    public init(code: String, name: String) {
        self.code = code
        self.name = name
    }
}

public class LanguageSupport {
    public static let supportedLanguages = [
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

// MARK: - Network Service
public class NetworkService {
    public static let shared = NetworkService()
    // For local testing: Use your computer's IP address
    // For production: Use "https://cursor-first-test.onrender.com"
    private let baseURL = "https://cursor-first-test.onrender.com" // Production deployment
    
    private init() {}
    
    public func translateText(
        text: String, 
        fromLanguage: String = "auto", 
        toLanguage: String = "en"
    ) async -> Result<String, Error> {
        guard let url = URL(string: "\(baseURL)/translate") else {
            let error = NSError(domain: "NetworkService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
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
        request.timeoutInterval = 30.0
        
        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "NetworkService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    return .failure(error)
                }
            }
            
            let translationResponse = try JSONDecoder().decode(TranslationResponse.self, from: data)
            return .success(translationResponse.translated_text)
            
        } catch {
            return .failure(error)
        }
    }
    
    public func translatePDF(
        pdfData: Data,
        filename: String,
        fromLanguage: String = "auto",
        toLanguage: String = "en"
    ) async -> Result<Data, Error> {
        guard let url = URL(string: "\(baseURL)/translate-pdf") else {
            let error = NSError(domain: "NetworkService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            return .failure(error)
        }
        
        // Create multipart form data request
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300.0
        
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
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "NetworkService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    return .failure(error)
                }
            }
            
            return .success(data)
            
        } catch {
            return .failure(error)
        }
    }
} 