import SwiftUI
import UniformTypeIdentifiers

struct PDFTranslationView: View {
    @StateObject private var pdfService = PDFTranslationService.shared
    @State private var sourceLang = "auto"
    @State private var targetLang = "en"
    @State private var showingFilePicker = false
    @State private var selectedPDFData: Data?
    @State private var selectedFileName = ""
    @State private var showingShareSheet = false
    @State private var translatedPDFData: Data?
    @State private var showError = false
    @State private var errorMessage = ""
    
    // UserDefaults keys for saving language preferences
    private let sourceLanguageKey = "PDFLastUsedSourceLanguage"
    private let targetLanguageKey = "PDFLastUsedTargetLanguage"
    
    // Language definitions matching ContentView
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
    
    // Function to get device language and map to supported language
    private func getDeviceLanguage() -> String {
        let deviceLanguage = Locale.current.languageCode ?? "en"
        
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
        
        sourceLang = savedSourceLang ?? "auto"
        
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
                    // Header with app logo - exact match to ContentView
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
                            ForEach(supportedLanguages, id: \.code) { language in
                                Button(language.name) {
                                    sourceLang = language.code
                                }
                            }
                        } label: {
                            HStack {
                                Text(sourceLang == "auto" ? "Detect" : (supportedLanguages.first(where: { $0.code == sourceLang })?.name ?? "Detect"))
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
                        .disabled(pdfService.isProcessing)
                        .frame(maxWidth: .infinity)
                        
                        // Arrow
                        Image(systemName: "arrow.right")
                            .foregroundColor(.blue)
                            .font(.caption)
                        
                        // Target Language
                        Menu {
                            ForEach(supportedLanguages, id: \.code) { language in
                                Button(language.name) {
                                    targetLang = language.code
                                }
                            }
                        } label: {
                            HStack {
                                Text(supportedLanguages.first(where: { $0.code == targetLang })?.name ?? "English")
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
                        .disabled(pdfService.isProcessing)
                        .frame(maxWidth: .infinity)
                    }
                    
                    // PDF Upload Area - replacing text input area
                    VStack(alignment: .leading, spacing: 8) {
                        if let _ = selectedPDFData {
                            // PDF Selected - Show file info
                            VStack(spacing: 16) {
                                HStack {
                                    Image("PDF-Icon")
                                        .resizable()
                                        .frame(width: 32, height: 32)
                                        .foregroundColor(.red)
                                    
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(selectedFileName.isEmpty ? "Document.pdf" : selectedFileName)
                                            .font(.headline)
                                            .foregroundColor(.primary)
                                        
                                        Text("PDF ready for translation")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    Spacer()
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(12)
                                
                                // Language Selection
                                HStack {
                                    Text("Translate to:")
                                        .font(.headline)
                                    
                                    Picker("Target Language", selection: $targetLang) {
                                        ForEach(supportedLanguages, id: \.code) { language in
                                            Text(language.name).tag(language.code)
                                        }
                                    }
                                    .pickerStyle(MenuPickerStyle())
                                    .frame(maxWidth: .infinity)
                                }
                                
                                // Translate Button
                                Button(action: translatePDF) {
                                    HStack {
                                        if pdfService.isProcessing {
                                            ProgressView()
                                                .scaleEffect(0.8)
                                        }
                                        Text(pdfService.isProcessing ? "Translating..." : "Translate PDF")
                                            .font(.headline)
                                    }
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 16)
                                    .background(pdfService.isProcessing ? Color.gray : Color.blue)
                                    .cornerRadius(12)
                                }
                                .disabled(pdfService.isProcessing)
                            }
                        } else {
                            // Upload area
                            VStack(spacing: 24) {
                                Image("PDF-Icon")
                                    .resizable()
                                    .frame(width: 64, height: 64)
                                    .foregroundColor(.gray)
                                
                                VStack(spacing: 8) {
                                    Text("Select PDF to Translate")
                                        .font(.title2)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)
                                    
                                    Text("Choose a PDF file from your device")
                                        .font(.body)
                                        .foregroundColor(.secondary)
                                        .multilineTextAlignment(.center)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 60)
                            .background(Color(.systemGray6))
                            .cornerRadius(16)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                showingFilePicker = true
                            }
                        }
                        
                        Spacer()
                        
                        // Full width Upload PDF button
                        if selectedPDFData == nil {
                            Button(action: { showingFilePicker = true }) {
                                HStack {
                                    Text("Select PDF File")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(Color.blue)
                                .cornerRadius(12)
                            }
                        }
                    }
                    
                    // Translation Results Area - matching ContentView output area
                    VStack(alignment: .leading, spacing: 8) {
                        // Removed separate share button as requested
                    }
                        
                        ZStack {
                            VStack {
                                if pdfService.isProcessing {
                                    VStack(spacing: 12) {
                                        ProgressView(value: pdfService.progress)
                                            .progressViewStyle(LinearProgressViewStyle())
                                            .frame(height: 6)
                                        
                                        Text("Translating PDF...")
                                            .font(.body)
                                            .foregroundColor(.secondary)
                                        
                                        Text("\(Int(pdfService.progress * 100))%")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        
                                        Button("Cancel") {
                                            pdfService.cancelCurrentRequest()
                                        }
                                        .foregroundColor(.red)
                                        .font(.caption)
                                    }
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    .padding()
                                } else if translatedPDFData != nil {
                                    VStack(spacing: 16) {
                                        Text("PDF translated")
                                            .font(.body)
                                            .foregroundColor(.black)
                                        
                                        Button(action: { showingShareSheet = true }) {
                                            HStack {
                                                Image(systemName: "square.and.arrow.down")
                                                Text("Save Translated PDF")
                                            }
                                            .font(.headline)
                                            .foregroundColor(.white)
                                            .padding(.horizontal, 30)
                                            .padding(.vertical, 12)
                                            .background(Color.blue)
                                            .cornerRadius(10)
                                        }
                                        
                                        Button("Translate Another PDF") {
                                            resetView()
                                        }
                                        .foregroundColor(.blue)
                                        .font(.caption)
                                    }
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    .padding()
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
                loadSavedLanguages()
            }
            .onChange(of: sourceLang) { _ in
                saveLanguagePreferences()
            }
            .onChange(of: targetLang) { _ in
                saveLanguagePreferences()
            }
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: [.pdf],
                allowsMultipleSelection: false
            ) { result in
                handleFileSelection(result)
            }
            .sheet(isPresented: $showingShareSheet) {
                if let pdfData = translatedPDFData {
                    ShareSheet(items: [createTempPDFURL(from: pdfData)])
                }
            }
            .alert("Translation Error", isPresented: $showError) {
                Button("OK") { showError = false }
            } message: {
                Text(errorMessage)
            }
    }
    
    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            
            // Start accessing security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                errorMessage = "Failed to access PDF file. Please try again."
                showError = true
                return
            }
            
            defer {
                // Always stop accessing the security-scoped resource
                url.stopAccessingSecurityScopedResource()
            }
            
            do {
                let data = try Data(contentsOf: url)
                
                // Validate PDF file
                guard isValidPDF(data: data) else {
                    errorMessage = "Invalid PDF file. Please select a valid PDF document."
                    showError = true
                    return
                }
                
                // Check file size (limit to 50MB)
                let maxSize = 50 * 1024 * 1024 // 50MB
                guard data.count <= maxSize else {
                    errorMessage = "PDF file is too large. Please select a file smaller than 50MB."
                    showError = true
                    return
                }
                
                selectedPDFData = data
                selectedFileName = url.lastPathComponent
                translatedPDFData = nil // Reset previous translation
                
                // Auto-start translation after successful upload
                translatePDF()
            } catch {
                errorMessage = "Failed to load PDF: \(error.localizedDescription)"
                showError = true
            }
            
        case .failure(let error):
            errorMessage = "File selection failed: \(error.localizedDescription)"
            showError = true
        }
    }
    
    private func isValidPDF(data: Data) -> Bool {
        // Check minimum size
        guard data.count >= 4 else {
            print("PDF validation failed: File too small (\(data.count) bytes)")
            return false
        }
        
        // Check PDF magic bytes - PDF files start with "%PDF-"
        let pdfHeader = Data([0x25, 0x50, 0x44, 0x46, 0x2D]) // "%PDF-"
        let fileHeader = data.prefix(5)
        
        guard fileHeader == pdfHeader else {
            print("PDF validation failed: Invalid header. Expected %PDF-, got: \(fileHeader.map { String(format: "%02x", $0) }.joined())")
            return false
        }
        
        // For now, let's be more permissive and just check for the basic PDF header
        // This should work for most valid PDF files
        print("PDF validation passed: Valid PDF header found")
        return true
    }
    
    private func translatePDF() {
        guard let pdfData = selectedPDFData else { return }
        
        Task {
            let result = await pdfService.translatePDF(
                pdfData: pdfData,
                filename: selectedFileName,
                fromLanguage: sourceLang,
                toLanguage: targetLang
            )
            
            await MainActor.run {
                switch result {
                case .success(let data):
                    translatedPDFData = data
                case .failure(let error):
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
    
    private func createTempPDFURL(from data: Data) -> URL {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("translated_\(selectedFileName)")
        try? data.write(to: tempURL)
        return tempURL
    }
    
    private func resetView() {
        selectedPDFData = nil
        selectedFileName = ""
        translatedPDFData = nil
        pdfService.progress = 0.0
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct PDFTranslationView_Previews: PreviewProvider {
    static var previews: some View {
        PDFTranslationView()
    }
} 