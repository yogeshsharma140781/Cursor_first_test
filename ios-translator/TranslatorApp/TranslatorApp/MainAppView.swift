import SwiftUI

struct MainAppView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            ContentView()
                .tabItem {
                    Image(systemName: "text.bubble")
                    Text("Text")
                }
                .tag(0)
            
            PDFTranslationView()
                .tabItem {
                    Image(systemName: "doc.text")
                    Text("PDF")
                }
                .tag(1)
        }
        .accentColor(.blue)
    }
}

struct MainAppView_Previews: PreviewProvider {
    static var previews: some View {
        MainAppView()
    }
} 