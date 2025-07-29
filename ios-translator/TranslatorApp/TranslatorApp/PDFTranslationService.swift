import Foundation
import UIKit
import UserNotifications

@MainActor
class PDFTranslationService: ObservableObject {
    static let shared = PDFTranslationService()
    
    // Update this URL to match your deployment
    // For local testing: Use your computer's IP address
    // For local testing: Use your computer's IP address
// For production: Use "https://cursor-first-test.onrender.com"
private let baseURL = "https://cursor-first-test.onrender.com" // Production deployment
    
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var progress: Double = 0.0
    
    private var currentTask: Task<Void, Never>?
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    private var currentTranslationID: String?
    
    // UserDefaults keys for persistence
    private let processingKey = "PDFTranslationProcessing"
    private let progressKey = "PDFTranslationProgress"
    private let translationIDKey = "PDFTranslationID"
    private let filenameKey = "PDFTranslationFilename"
    private let sourceLangKey = "PDFTranslationSourceLang"
    private let targetLangKey = "PDFTranslationTargetLang"
    
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
    
    func translatePDF(
        pdfData: Data,
        filename: String,
        fromLanguage: String = "auto",
        toLanguage: String = "en"
    ) async -> Result<Data, Error> {
        // Cancel any existing request
        currentTask?.cancel()
        
        // Generate unique translation ID
        let translationID = UUID().uuidString
        currentTranslationID = translationID
        
        // Save translation state for persistence
        saveTranslationState(
            isProcessing: true,
            progress: 0.0,
            translationID: translationID,
            filename: filename,
            sourceLang: fromLanguage,
            targetLang: toLanguage
        )
        
        isProcessing = true
        errorMessage = nil
        progress = 0.0
        
        // Begin background task
        beginBackgroundTask()
        
        defer {
            // End background task
            endBackgroundTask()
            
            // Clear persistence if not backgrounded
            if !isProcessing {
                clearTranslationState()
            }
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
        
        // Add translation ID parameter
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"translation_id\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(translationID)\r\n".data(using: .utf8)!)
        
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
            updateProgress(0.3)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            progress = 0.8 // Processing complete
            updateProgress(0.8)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    // Reset processing state on HTTP error
                    isProcessing = false
                    clearTranslationState()
                    endBackgroundTask()
                    
                    let error = NSError(domain: "PDFTranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    errorMessage = "PDF translation failed. Please try again."
                    return .failure(error)
                }
            }
            
            progress = 1.0 // Complete
            updateProgress(1.0)
            
            // Reset processing state
            isProcessing = false
            clearTranslationState()
            endBackgroundTask()
            
            // Send completion notification
            sendCompletionNotification(filename: filename, success: true)
            
            return .success(data)
            
        } catch {
            if error.localizedDescription.contains("cancelled") {
                // Request was cancelled, don't show error
                return .failure(error)
            }
            
            // Reset processing state on error
            isProcessing = false
            clearTranslationState()
            endBackgroundTask()
            
            errorMessage = "Network error. Please check your connection."
            
            // Send error notification
            sendCompletionNotification(filename: filename, success: false, error: error.localizedDescription)
            
            return .failure(error)
        }
    }
    
    func cancelCurrentRequest() {
        currentTask?.cancel()
        isProcessing = false
        progress = 0.0
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
        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "PDFTranslation") { [weak self] in
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
        progress: Double,
        translationID: String,
        filename: String,
        sourceLang: String,
        targetLang: String
    ) {
        UserDefaults.standard.set(isProcessing, forKey: processingKey)
        UserDefaults.standard.set(progress, forKey: progressKey)
        UserDefaults.standard.set(translationID, forKey: translationIDKey)
        UserDefaults.standard.set(filename, forKey: filenameKey)
        UserDefaults.standard.set(sourceLang, forKey: sourceLangKey)
        UserDefaults.standard.set(targetLang, forKey: targetLangKey)
    }
    
    private func updateProgress(_ newProgress: Double) {
        progress = newProgress
        UserDefaults.standard.set(newProgress, forKey: progressKey)
    }
    
    private func clearTranslationState() {
        UserDefaults.standard.removeObject(forKey: processingKey)
        UserDefaults.standard.removeObject(forKey: progressKey)
        UserDefaults.standard.removeObject(forKey: translationIDKey)
        UserDefaults.standard.removeObject(forKey: filenameKey)
        UserDefaults.standard.removeObject(forKey: sourceLangKey)
        UserDefaults.standard.removeObject(forKey: targetLangKey)
    }
    
    private func checkForOngoingTranslations() {
        let wasProcessing = UserDefaults.standard.bool(forKey: processingKey)
        if wasProcessing {
            let savedProgress = UserDefaults.standard.double(forKey: progressKey)
            let savedTranslationID = UserDefaults.standard.string(forKey: translationIDKey)
            
            // Restore state
            isProcessing = true
            progress = savedProgress
            currentTranslationID = savedTranslationID
            
            // Show notification that translation is still in progress
            sendInProgressNotification()
        }
    }
    
    // MARK: - Notifications
    
    private func sendCompletionNotification(filename: String, success: Bool, error: String? = nil) {
        let content = UNMutableNotificationContent()
        content.title = success ? "Translation Complete" : "Translation Failed"
        content.body = success ? 
            "Your PDF '\(filename)' has been translated successfully." :
            "Failed to translate '\(filename)'. \(error ?? "Please try again.")"
        content.sound = success ? .default : .default
        
        let request = UNNotificationRequest(
            identifier: "translation-\(currentTranslationID ?? UUID().uuidString)",
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
        content.body = "Your PDF translation is still running in the background."
        content.sound = nil
        
        let request = UNNotificationRequest(
            identifier: "translation-in-progress",
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Notification error: \(error)")
            }
        }
    }
    
    // Helper function to save translated PDF to device
    func savePDFToDocuments(data: Data, filename: String) -> URL? {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        
        let fileURL = documentsDirectory.appendingPathComponent("translated_\(filename)")
        
        do {
            try data.write(to: fileURL)
            return fileURL
        } catch {
            print("Error saving PDF: \(error)")
            return nil
        }
    }
} 