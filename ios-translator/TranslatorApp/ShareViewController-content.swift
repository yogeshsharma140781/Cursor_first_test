import UIKit
import Social
import SwiftUI
import UniformTypeIdentifiers

class ShareViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Extract the shared text
        extractSharedText { [weak self] sharedText in
            DispatchQueue.main.async {
                self?.setupTranslationView(with: sharedText)
            }
        }
    }
    
    private func extractSharedText(completion: @escaping (String) -> Void) {
        guard let extensionItem = extensionContext?.inputItems.first as? NSExtensionItem,
              let itemProvider = extensionItem.attachments?.first else {
            completion("No text found")
            return
        }
        
        // Try to get plain text first
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.plainText.identifier, options: nil) { item, error in
                if let text = item as? String {
                    completion(text)
                } else if let data = item as? Data, let text = String(data: data, encoding: .utf8) {
                    completion(text)
                } else {
                    completion("Unable to extract text")
                }
            }
        } else {
            completion("No text content found")
        }
    }
    
    private func setupTranslationView(with text: String) {
        let translationView = TranslationView(
            originalText: text,
            onTranslationComplete: { [weak self] in
                self?.extensionContext?.completeRequest(returningItems: nil, completionHandler: nil)
            },
            onCancel: { [weak self] in
                self?.extensionContext?.cancelRequest(withError: NSError(domain: "UserCancelled", code: 0, userInfo: nil))
            }
        )
        
        let hostingController = UIHostingController(rootView: translationView)
        addChild(hostingController)
        view.addSubview(hostingController.view)
        
        hostingController.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hostingController.view.topAnchor.constraint(equalTo: view.topAnchor),
            hostingController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        
        hostingController.didMove(toParent: self)
    }
}

struct TranslationView: View {
    let originalText: String
    let onTranslationComplete: () -> Void
    let onCancel: () -> Void
    
    @State private var selectedLanguage = "en"
    @State private var translatedText = ""
    @State private var isTranslating = false
    @State private var showError = false
    @State private var errorMessage = ""
    
    @StateObject private var translationService = TranslationService.shared
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Original text section
                VStack(alignment: .leading, spacing: 8) {
                    Text("Original Text")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    ScrollView {
                        Text(originalText)
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .frame(maxHeight: 120)
                }
                
                // Language selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Translate to")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Picker("Target Language", selection: $selectedLanguage) {
                        ForEach(TranslationService.supportedLanguages, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                }
                
                // Translation button
                Button(action: performTranslation) {
                    HStack {
                        if isTranslating {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isTranslating ? "Translating..." : "Translate")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isTranslating)
                
                // Translated text section
                if !translatedText.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Translation")
                                .font(.headline)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            Button(action: copyTranslation) {
                                HStack {
                                    Image(systemName: "doc.on.doc")
                                    Text("Copy")
                                }
                                .font(.caption)
                                .foregroundColor(.blue)
                            }
                        }
                        
                        ScrollView {
                            Text(translatedText)
                                .font(.body)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(8)
                        }
                        .frame(maxHeight: 120)
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Translate with AI")
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
            // Auto-translate on appear
            performTranslation()
        }
    }
    
    private func performTranslation() {
        guard !originalText.isEmpty else { return }
        
        isTranslating = true
        
        Task {
            let result = await translationService.translate(
                text: originalText,
                toLanguage: selectedLanguage
            )
            
            DispatchQueue.main.async {
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
    }
    
    private func copyTranslation() {
        UIPasteboard.general.string = translatedText
    }
} 