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
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Language selection - clean and minimal
                HStack {
                    Picker("Target Language", selection: $selectedLanguage) {
                        ForEach(TranslationService.supportedLanguages, id: \.code) { language in
                            Text(language.name).tag(language.code)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedLanguage) { _ in
                        performTranslation()
                    }
                    
                    Spacer()
                }
                
                // Full-size translated text section
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
            .navigationTitle("Translay AI")
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
            let result = await TranslationService.shared.translate(
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
    
    private func copyTranslation() {
        UIPasteboard.general.string = translatedText
    }
} 