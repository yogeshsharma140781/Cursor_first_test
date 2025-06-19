import UIKit
import Social
import SwiftUI
import UniformTypeIdentifiers

class ShareViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Check if we have a PDF or text content
        checkSharedContent { [weak self] contentType in
            DispatchQueue.main.async {
                switch contentType {
                case .text(let text):
                    self?.setupTextTranslationView(with: text)
                case .pdf(let data, let filename):
                    self?.setupPDFTranslationView(with: data, filename: filename)
                case .none:
                    self?.setupErrorView(message: "No supported content found")
                }
            }
        }
    }
    
    enum SharedContentType {
        case text(String)
        case pdf(Data, String)
        case none
    }
    
    private func checkSharedContent(completion: @escaping (SharedContentType) -> Void) {
        guard let extensionItem = extensionContext?.inputItems.first as? NSExtensionItem,
              let itemProvider = extensionItem.attachments?.first else {
            completion(.none)
            return
        }
        
        // Check for PDF files first
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.pdf.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.pdf.identifier, options: nil) { [weak self] item, error in
                if let url = item as? URL {
                    self?.loadPDFFromURL(url: url, completion: completion)
                } else if let data = item as? Data {
                    completion(.pdf(data, "shared_document.pdf"))
                } else {
                    // If PDF loading fails, try text
                    self?.extractTextContent(from: itemProvider, completion: completion)
                }
            }
        } else {
            // Try to extract text content
            extractTextContent(from: itemProvider, completion: completion)
        }
    }
    
    private func loadPDFFromURL(url: URL, completion: @escaping (SharedContentType) -> Void) {
        // Start accessing security-scoped resource
        guard url.startAccessingSecurityScopedResource() else {
            completion(.none)
            return
        }
        
        defer {
            url.stopAccessingSecurityScopedResource()
        }
        
        do {
            let data = try Data(contentsOf: url)
            let filename = url.lastPathComponent
            completion(.pdf(data, filename))
        } catch {
            print("Failed to load PDF from URL: \(error)")
            completion(.none)
        }
    }
    
    private func extractTextContent(from itemProvider: NSItemProvider, completion: @escaping (SharedContentType) -> Void) {
        // Try to get plain text first
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.plainText.identifier, options: nil) { item, error in
                if let text = item as? String {
                    completion(.text(text))
                } else if let data = item as? Data, let text = String(data: data, encoding: .utf8) {
                    completion(.text(text))
                } else {
                    completion(.none)
                }
            }
        } else if itemProvider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { item, error in
                if let url = item as? URL {
                    completion(.text(url.absoluteString))
                } else {
                    completion(.none)
                }
            }
        } else {
            completion(.none)
        }
    }
    
    private func setupTextTranslationView(with text: String) {
        let translationView = SimpleTranslationView(
            originalText: text,
            onTranslationComplete: { [weak self] in
                self?.extensionContext?.completeRequest(returningItems: nil, completionHandler: nil)
            },
            onCancel: { [weak self] in
                self?.extensionContext?.cancelRequest(withError: NSError(domain: "UserCancelled", code: 0, userInfo: nil))
            }
        )
        
        setupHostingController(with: translationView)
    }
    
    private func setupPDFTranslationView(with pdfData: Data, filename: String) {
        let pdfTranslationView = PDFTranslationExtensionView(
            pdfData: pdfData,
            filename: filename,
            onTranslationComplete: { [weak self] translatedData in
                // Share the translated PDF
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("translated_\(filename)")
                try? translatedData.write(to: tempURL)
                
                let activityViewController = UIActivityViewController(activityItems: [tempURL], applicationActivities: nil)
                self?.present(activityViewController, animated: true) {
                    self?.extensionContext?.completeRequest(returningItems: nil, completionHandler: nil)
                }
            },
            onCancel: { [weak self] in
                self?.extensionContext?.cancelRequest(withError: NSError(domain: "UserCancelled", code: 0, userInfo: nil))
            }
        )
        
        setupHostingController(with: pdfTranslationView)
    }
    
    private func setupErrorView(message: String) {
        let errorView = ErrorView(
            message: message,
            onDismiss: { [weak self] in
                self?.extensionContext?.cancelRequest(withError: NSError(domain: "NoContent", code: 0, userInfo: nil))
            }
        )
        
        setupHostingController(with: errorView)
    }
    
    private func setupHostingController<T: View>(with view: T) {
        let hostingController = UIHostingController(rootView: view)
        addChild(hostingController)
        self.view.addSubview(hostingController.view)
        
        hostingController.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hostingController.view.topAnchor.constraint(equalTo: self.view.topAnchor),
            hostingController.view.leadingAnchor.constraint(equalTo: self.view.leadingAnchor),
            hostingController.view.trailingAnchor.constraint(equalTo: self.view.trailingAnchor),
            hostingController.view.bottomAnchor.constraint(equalTo: self.view.bottomAnchor)
        ])
        
        hostingController.didMove(toParent: self)
    }
}

struct SimpleTranslationView: View {
    let originalText: String
    let onTranslationComplete: () -> Void
    let onCancel: () -> Void
    
    @State private var selectedLanguage = "en"
    @State private var translatedText = ""
    @State private var isTranslating = false
    @State private var showError = false
    @State private var errorMessage = ""
    
    // Define supported languages locally for the extension
    private let supportedLanguages = [
        (code: "en", name: "English"),
        (code: "ar", name: "Arabic"),
        (code: "zh-cn", name: "Chinese (Simplified)"),
        (code: "zh-tw", name: "Chinese (Traditional)"),
        (code: "nl", name: "Dutch"),
        (code: "fr", name: "French"),
        (code: "de", name: "German"),
        (code: "hi", name: "Hindi"),
        (code: "it", name: "Italian"),
        (code: "ja", name: "Japanese"),
        (code: "ko", name: "Korean"),
        (code: "pl", name: "Polish"),
        (code: "pt", name: "Portuguese"),
        (code: "ru", name: "Russian"),
        (code: "es", name: "Spanish"),
        (code: "tr", name: "Turkish"),
        (code: "uk", name: "Ukrainian"),
        (code: "vi", name: "Vietnamese")
    ]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Language selection
                HStack {
                    Picker("Target Language", selection: $selectedLanguage) {
                        ForEach(supportedLanguages, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedLanguage) { _ in
                        performTranslation()
                    }
                    
                    Spacer()
                }
                
                // Translated text section
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Translation")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        if !translatedText.isEmpty {
                            Button(action: copyTranslation) {
                                HStack {
                                    Image(systemName: "doc.on.doc")
                                    Text("Copy")
                                }
                                .font(.caption)
                                .foregroundColor(.blue)
                            }
                        }
                    }
                    
                    ScrollView {
                        if isTranslating {
                            VStack {
                                ProgressView()
                                    .scaleEffect(1.2)
                                    .padding()
                                Text("Translating...")
                                    .font(.body)
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .padding(40)
                        } else if translatedText.isEmpty {
                            Text("Translation will appear here")
                                .font(.body)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .padding(40)
                        } else {
                            Text(translatedText)
                                .font(.body)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                        }
                    }
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(8)
                    .frame(maxHeight: .infinity)
                }
                
                Spacer(minLength: 0)
            }
            .padding()
            .navigationTitle("Translate")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel", action: onCancel),
                trailing: !translatedText.isEmpty ? Button("Done", action: onTranslationComplete) : nil
            )
        }
        .alert("Translation Error", isPresented: $showError) {
            Button("OK") { showError = false }
        } message: {
            Text(errorMessage)
        }
        .onAppear {
            performTranslation()
        }
    }
    
    private func performTranslation() {
        guard !originalText.isEmpty else { return }
        
        isTranslating = true
        translatedText = ""
        
        Task { @MainActor in
            let result = await translateText(
                text: originalText,
                toLanguage: selectedLanguage
            )
            
            isTranslating = false
            
            switch result {
            case .success(let translation):
                translatedText = translation
            case .failure(let error):
                errorMessage = error.localizedDescription
                showError = true
            }
        }
    }
    
    private func translateText(text: String, fromLanguage: String = "auto", toLanguage: String = "en") async -> Result<String, Error> {
        guard let url = URL(string: "https://cursor-first-test.onrender.com/translate") else {
            let error = NSError(domain: "TranslationService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            return .failure(error)
        }
        
        let requestBody = [
            "text": text,
            "source_lang": fromLanguage,
            "target_lang": toLanguage
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                guard httpResponse.statusCode == 200 else {
                    let error = NSError(domain: "TranslationService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error: \(httpResponse.statusCode)"])
                    return .failure(error)
                }
            }
            
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let translation = json["translation"] as? String {
                return .success(translation)
            } else {
                let error = NSError(domain: "TranslationService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
                return .failure(error)
            }
            
        } catch {
            return .failure(error)
        }
    }
    
    private func copyTranslation() {
        UIPasteboard.general.string = translatedText
    }
}

// PDF Translation View for Share Extension
struct PDFTranslationExtensionView: View {
    let pdfData: Data
    let filename: String
    let onTranslationComplete: (Data) -> Void
    let onCancel: () -> Void
    
    @State private var selectedLanguage = "en"
    @State private var isTranslating = false
    @State private var showError = false
    @State private var errorMessage = ""
    @StateObject private var pdfService = PDFTranslationService.shared
    
    // Define supported languages locally for the extension
    private let supportedLanguages = [
        (code: "en", name: "English"),
        (code: "ar", name: "Arabic"),
        (code: "zh-cn", name: "Chinese (Simplified)"),
        (code: "zh-tw", name: "Chinese (Traditional)"),
        (code: "nl", name: "Dutch"),
        (code: "fr", name: "French"),
        (code: "de", name: "German"),
        (code: "hi", name: "Hindi"),
        (code: "it", name: "Italian"),
        (code: "ja", name: "Japanese"),
        (code: "ko", name: "Korean"),
        (code: "pl", name: "Polish"),
        (code: "pt", name: "Portuguese"),
        (code: "ru", name: "Russian"),
        (code: "es", name: "Spanish"),
        (code: "tr", name: "Turkish"),
        (code: "uk", name: "Ukrainian"),
        (code: "vi", name: "Vietnamese")
    ]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // PDF Info
                VStack(spacing: 12) {
                    Image("PDF-Icon")
                        .resizable()
                        .frame(width: 48, height: 48)
                        .foregroundColor(.red)
                    
                    Text(filename)
                        .font(.headline)
                        .multilineTextAlignment(.center)
                    
                    Text("PDF ready for translation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                
                // Language selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Translate to:")
                        .font(.headline)
                    
                    Picker("Target Language", selection: $selectedLanguage) {
                        ForEach(supportedLanguages, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                // Translation button or progress
                if isTranslating {
                    VStack(spacing: 16) {
                        ProgressView(value: pdfService.progress)
                            .progressViewStyle(LinearProgressViewStyle())
                        
                        Text("Translating PDF...")
                            .font(.body)
                            .foregroundColor(.secondary)
                        
                        Text("\(Int(pdfService.progress * 100))%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                } else {
                    Button("Translate PDF") {
                        translatePDF()
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color.blue)
                    .cornerRadius(12)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("PDF Translation")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel", action: onCancel)
            )
        }
        .alert("Translation Error", isPresented: $showError) {
            Button("OK") { showError = false }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func translatePDF() {
        isTranslating = true
        
        Task {
            let result = await pdfService.translatePDF(
                pdfData: pdfData,
                filename: filename,
                fromLanguage: "auto",
                toLanguage: selectedLanguage
            )
            
            await MainActor.run {
                isTranslating = false
                
                switch result {
                case .success(let translatedData):
                    onTranslationComplete(translatedData)
                case .failure(let error):
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
}

// Error View for Share Extension
struct ErrorView: View {
    let message: String
    let onDismiss: () -> Void
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 48))
                    .foregroundColor(.orange)
                
                Text("Unable to Process")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text(message)
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                
                Button("OK") {
                    onDismiss()
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.blue)
                .cornerRadius(12)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Error")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
} 
