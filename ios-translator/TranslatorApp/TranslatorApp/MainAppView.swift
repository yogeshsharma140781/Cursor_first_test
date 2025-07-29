import SwiftUI

struct MainAppView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Main content area
            Group {
                if selectedTab == 0 {
                    ContentView()
                } else {
                    PDFTranslationView()
                }
            }
            
            // Custom tab bar
            HStack {
                // Text tab
                Button(action: {
                    selectedTab = 0
                }) {
                    Image(selectedTab == 0 ? "SelectedText" : "UnselectedText")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 60, height: 60)
                }
                .frame(maxWidth: .infinity)
                
                // PDF tab
                Button(action: {
                    selectedTab = 1
                }) {
                    Image(selectedTab == 1 ? "SelectedPDF" : "UnselectedPDF")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 60, height: 60)
                }
                .frame(maxWidth: .infinity)
            }
            .padding(.top, 8)
            .padding(.bottom, 12)
            .background(Color(.systemBackground))
        }
        .ignoresSafeArea(.keyboard, edges: .bottom)
    }
}

struct MainAppView_Previews: PreviewProvider {
    static var previews: some View {
        MainAppView()
    }
} 