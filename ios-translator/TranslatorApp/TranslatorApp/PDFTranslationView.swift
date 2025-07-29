import SwiftUI
import UniformTypeIdentifiers
import PDFKit
import WebKit

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
    @State private var showFullScreenPreview = false
    
    // UserDefaults keys for saving language preferences
    private let sourceLanguageKey = "PDFLastUsedSourceLanguage"
    private let targetLanguageKey = "PDFLastUsedTargetLanguage"
    
    // Consistent blue color to match buttons
    private let appBlue = Color.blue
    
    // Language definitions matching ContentView
    private let supportedLanguages = [
        (code: "en", name: "English"),
        (code: "nl", name: "Dutch"),
        (code: "fr", name: "French"),
        (code: "de", name: "German"),
        (code: "it", name: "Italian"),
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
                            ForEach(supportedLanguages, id: \.code) { language in
                                Button(language.name) {
                                    sourceLang = language.code
                                }
                            }
                        } label: {
                            HStack {
                                Text(sourceLang == "auto" ? "Detect" : (supportedLanguages.first(where: { $0.code == sourceLang })?.name ?? "Detect"))
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
                        .disabled(pdfService.isProcessing)
                        .frame(maxWidth: .infinity)
                        
                        // Arrow
                        Image(systemName: "arrow.right")
                            .foregroundColor(.secondary)
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
                        .disabled(pdfService.isProcessing)
                        .frame(maxWidth: .infinity)
                    }
                    
                    // PDF Upload Area - replacing text input area
                    VStack(alignment: .leading, spacing: 8) {
                        if let _ = selectedPDFData {
                            // PDF Selected - Show file info and translation status
                            VStack(spacing: 16) {
                                HStack {
                                    Image("PDFdocument")
                                        .resizable()
                                        .frame(width: 48, height: 48)
                                        .foregroundColor(.primary)
                                    
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(selectedFileName.isEmpty ? "Document.pdf" : selectedFileName)
                                            .font(.headline)
                                            .foregroundColor(.primary)
                                        
                                                                                // Show translation status
                                        if pdfService.isProcessing {
                                            HStack(spacing: 8) {
                                                ProgressView()
                                                    .scaleEffect(0.7)
                                                Text("Translating...")
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                            }
                                        } else if translatedPDFData != nil {
                                            Text("Translation complete")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        } else {
                                            Text("Starting translation...")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    
                                    Spacer()
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(12)
                            }
                        } else {
                            // Upload area
                            VStack(spacing: 24) {
                                Image(systemName: "doc.badge.arrow.up")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                
                                VStack(spacing: 8) {
                                    Text("PDF Translator")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)
                                    
                                    Text("Upload PDF to translate")
                                        .font(.caption)
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
                        
                        // Center animation area - show GIF when translating
                        if pdfService.isProcessing {
                            VStack(spacing: 20) {
                                // Center the GIF animation
                                HStack {
                                    Spacer()
                                    
                                    // GIF animation with fallback
                                    ThemeAwareGIFView()
                                        .frame(width: 80, height: 80)
                                        .clipped()
                                    
                                    Spacer()
                                }
                                
                                // Cancel button
                                Button(action: {
                                    pdfService.cancelTranslation()
                                }) {
                                    HStack {
                                        Image(systemName: "xmark.circle.fill")
                                        Text("Cancel Translation")
                                    }
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 24)
                                    .padding(.vertical, 12)
                                    .background(Color.red)
                                    .cornerRadius(10)
                                }
                            }
                            .padding(.vertical, 20)
                        } else {
                            Spacer()
                        }
                        
                        // Full width Upload PDF button
                        if selectedPDFData == nil {
                            Button(action: { showingFilePicker = true }) {
                                HStack {
                                    Text("Upload PDF")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(appBlue)
                                .cornerRadius(12)
                            }
                        }
                    }
                    
                    // Show success message and action buttons when translation is complete
                    if translatedPDFData != nil {
                        VStack(spacing: 16) {
                            Text("PDF translated successfully!")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            HStack(spacing: 12) {
                                Button(action: { showFullScreenPreview = true }) {
                                    HStack {
                                        Image(systemName: "eye")
                                        Text("Preview")
                                    }
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 20)
                                    .padding(.vertical, 12)
                                    .background(appBlue)
                                    .cornerRadius(10)
                                }
                                
                                Button(action: { showingShareSheet = true }) {
                                    HStack {
                                        Image(systemName: "square.and.arrow.down")
                                        Text("Save")
                                    }
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 20)
                                    .padding(.vertical, 12)
                                    .background(appBlue)
                                    .cornerRadius(10)
                                }
                            }
                        }
                        .padding()
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
            .fullScreenCover(isPresented: $showFullScreenPreview) {
                if let translatedData = translatedPDFData, let originalData = selectedPDFData {
                    PDFPreviewView(
                        originalPdfData: originalData,
                        translatedPdfData: translatedData,
                        isPresented: $showFullScreenPreview
                    ) {
                        resetView()
                    }
                }
            }
            .alert("Translation Error", isPresented: $showError) {
                Button("OK") { showError = false }
            } message: {
                Text(errorMessage)
            }
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
                    // Automatically show preview when translation is complete
                    showFullScreenPreview = true
                case .failure(let error):
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
    
    // MARK: - App State Handling
    
    private func handleAppStateChange() {
        // Check if there's an ongoing translation when app becomes active
        if pdfService.isProcessing {
            // Translation is still in progress, show appropriate UI
            // The service will handle notifications
        }
    }
    
    private func handleBackgroundTask() {
        // This will be called when the app is about to go to background
        // The PDFTranslationService will handle the background task
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

struct PDFPreviewView: View {
    let originalPdfData: Data
    let translatedPdfData: Data
    @Binding var isPresented: Bool
    @State private var showingShareSheet = false
    @State private var showingOriginal = false
    let onDismiss: () -> Void
    
    private var currentPdfData: Data {
        return showingOriginal ? originalPdfData : translatedPdfData
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Top control bar with download, toggle, and done buttons
            HStack {
                // Download button
                Button(action: { showingShareSheet = true }) {
                    Image("DownloadIcon")
                        .renderingMode(.template)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 28, height: 28)
                        .foregroundColor(.blue)
                }
                
                Spacer()
                
                // Toggle control
                Picker("PDF Version", selection: $showingOriginal) {
                    Text("Translated").tag(false)
                    Text("Original").tag(true)
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(width: 200)
                
                Spacer()
                
                // Done button
                Button("Done") {
                    isPresented = false
                    onDismiss()
                }
                .foregroundColor(.blue)
                .font(.headline)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.systemBackground))
            .overlay(
                Rectangle()
                    .frame(height: 0.5)
                    .foregroundColor(Color(.separator)),
                alignment: .bottom
            )
            
            // PDF View
            PDFKitView(data: currentPdfData)
                .id(showingOriginal) // Force refresh when toggle changes
        }
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(items: [createTempPDFURL(from: currentPdfData)])
        }
    }
    
    private func createTempPDFURL(from data: Data) -> URL {
        let fileName = showingOriginal ? "original_document.pdf" : "translated_document.pdf"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        try? data.write(to: tempURL)
        return tempURL
    }
}

struct PDFKitView: UIViewRepresentable {
    let data: Data
    
    func makeUIView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.document = PDFDocument(data: data)
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        return pdfView
    }
    
    func updateUIView(_ uiView: PDFView, context: Context) {
        // No updates needed
    }
}

struct ThemeAwareGIFView: View {
    @State private var gifLoaded = false
    
    var body: some View {
        ZStack {
            // Show spinner while GIF is loading
            if !gifLoaded {
                ProgressView()
                    .scaleEffect(1.5)
                    .progressViewStyle(CircularProgressViewStyle(tint: .blue))
            }
            
            // Use single animation for both themes
            GIFView(gifName: "TranslatingAnimation") { loaded in
                DispatchQueue.main.async {
                    gifLoaded = loaded
                }
            }
        }
    }
}

struct GIFView: UIViewRepresentable {
    let gifName: String
    let onLoadComplete: ((Bool) -> Void)?
    
    init(gifName: String, onLoadComplete: ((Bool) -> Void)? = nil) {
        self.gifName = gifName
        self.onLoadComplete = onLoadComplete
    }
    
    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.backgroundColor = UIColor.clear
        webView.isOpaque = false
        webView.scrollView.isScrollEnabled = false
        webView.navigationDelegate = context.coordinator
        
        // Load the GIF animation
        if let gifData = loadGIFData() {
            let htmlString = """
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
            </head>
            <body style="margin:0; padding:0; background:transparent; display:flex; justify-content:center; align-items:center; width:100vw; height:100vh;">
            <img src="data:image/gif;base64,\(gifData.base64EncodedString())" 
                 style="max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain;" 
                 onload="document.title='loaded'" />
            </body>
            </html>
            """
            webView.loadHTMLString(htmlString, baseURL: nil)
        } else {
            onLoadComplete?(false)
        }
        
        return webView
    }
    
    func updateUIView(_ uiView: WKWebView, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        let parent: GIFView
        
        init(_ parent: GIFView) {
            self.parent = parent
        }
        
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            // Check if image loaded successfully
            webView.evaluateJavaScript("document.title") { result, error in
                let loaded = (result as? String) == "loaded"
                self.parent.onLoadComplete?(loaded)
            }
        }
    }
    
    private func loadGIFData() -> Data? {
        // Try loading from NSDataAsset first (Assets.xcassets)
        if let gifData = NSDataAsset(name: gifName)?.data {
            return gifData
        }
        
        // Try loading from bundle
        if let gifPath = Bundle.main.path(forResource: gifName, ofType: "gif"),
           let gifData = NSData(contentsOfFile: gifPath) {
            return gifData as Data
        }
        
        // Try loading without extension
        if let gifPath = Bundle.main.path(forResource: gifName, ofType: nil),
           let gifData = NSData(contentsOfFile: gifPath) {
            return gifData as Data
        }
        
        return nil
    }
}

struct PDFTranslationView_Previews: PreviewProvider {
    static var previews: some View {
        PDFTranslationView()
    }
} 