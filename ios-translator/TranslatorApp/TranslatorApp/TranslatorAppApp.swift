//
//  TranslatorAppApp.swift
//  TranslatorApp
//
//  Created by Yogesh Sharma on 15/06/2025.
//

import SwiftUI

@main
struct TranslatorAppApp: App {
    @StateObject private var pdfService = PDFTranslationService.shared
    @StateObject private var textService = TranslationService.shared
    
    var body: some Scene {
        WindowGroup {
            MainAppView()
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
                    // App became active - check for ongoing translations
                    handleAppDidBecomeActive()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
                    // App will become inactive - prepare for background
                    handleAppWillResignActive()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
                    // App entered background
                    handleAppDidEnterBackground()
                }
        }
    }
    
    private func handleAppDidBecomeActive() {
        // App became active - check for ongoing translations
        // The services will handle their own state restoration
        print("App became active - checking for ongoing translations")
    }
    
    private func handleAppWillResignActive() {
        // App will become inactive - ensure background tasks are properly set up
        print("App will resign active - setting up background tasks")
    }
    
    private func handleAppDidEnterBackground() {
        // App entered background - ensure translations continue
        print("App entered background - translations will continue")
    }
}
