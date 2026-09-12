//
//  ConversationScreen.swift
//  iosApp
//
//  Created by Juan Silva on 28/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

import SwiftUI

struct GemmaConversationScreen: View {
    private struct Constants {
        static let scrollDelayInSeconds = 0.05
        static let messageFieldPlaceHolder = "Message..."
        static let newChatSystemSymbolName = "square.and.pencil"
        static let navigationTitle = "Chat with your LLM here"
    }
    
    @EnvironmentObject
    var viewModel: GemmaConversationViewModel
    
    @State
    private var currentUserPrompt = ""
    
    private enum FocusedField: Hashable {
        case message
    }
    
    @FocusState
    private var focusedField: FocusedField?
    
    private var messageList: some View {
        List {
            ForEach(viewModel.messages) { message in
                GemmaMessageView(message: message)
            }
        }
        .listStyle(.plain)
    }
    
    private var inputField: some View {
        TextField(Constants.messageFieldPlaceHolder, text: $currentUserPrompt)
            .focused($focusedField, equals: .message)
            .onSubmit {
                sendMessage()
            }
            .submitLabel(.send)
            .disabled(viewModel.busy)
            .padding()
    }
    
    var body: some View {
        NavigationView {
            if viewModel.isLoading {
                ProgressView("Loading...")
                    .progressViewStyle(CircularProgressViewStyle())
            } else {
                VStack {
                    
                    if #available(iOS 17.0, *) {
                        ScrollViewReader { scrollViewProxy in
                            messageList
                                .onChange(of: viewModel.messages) { _, newValue in
                                    scrollToBottom(proxy: scrollViewProxy)
                                }
                                .onTapGesture {
                                    dismissKeyboard()
                                }
                        }
                    } else {
                        messageList
                            .onTapGesture {
                                dismissKeyboard()
                            }
                    }
                    
                    inputField
                }
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button(action: viewModel.startNewChat) {
                            Image(systemName: Constants.newChatSystemSymbolName)
                        }
                        .disabled(viewModel.busy)
                    }
                }
                .alert(error: $viewModel.error)
                .navigationTitle(Constants.navigationTitle)
                .onAppear {
                    focusedField = .message
                }
                // Add a tap gesture to the entire view
                .contentShape(Rectangle())
                .onTapGesture {
                    dismissKeyboard()
                }
            }
        }
    }
    
    private func dismissKeyboard() {
        focusedField = nil
    }
    
    private func sendMessage() {
        guard !currentUserPrompt.isEmpty else {
            return
        }
        let prompt = currentUserPrompt
        currentUserPrompt = ""
        viewModel.sendMessage(prompt)
    }
    
    @MainActor
    private func scrollToBottom(proxy: ScrollViewProxy) {
        Task {
            guard let lastMessage = viewModel.messages.last else { return }
            if #available(iOS 16.0, *) {
                try await Task.sleep(for: .seconds(Constants.scrollDelayInSeconds))
            } else {
                // Fallback on earlier versions
            }
            withAnimation {
                proxy.scrollTo(lastMessage.id, anchor: .bottom)
            }
            focusedField = .message
        }
    }
}
    
/// View that displays a message.
struct GemmaMessageView: View {
  private struct Constants {
    static let textMessagePadding: CGFloat = 10.0
    static let foregroundColor = Color(red: 0.0, green: 0.0, blue: 0.0)
    static let systemMessageBackgroundColor = Color(white: 0.9231)
    static let userMessageBackgroundColor = Color(red: 0.8627, green: 0.9725, blue: 0.7764)
    static let messageBackgroundCornerRadius: CGFloat = 16.0
  }
  /// Message to be displayed.
  var message: GemmaChatMessage

  var body: some View {
    HStack {
      if message.participant == .user {
        Spacer()
      }
      Text(message.text)
        .padding(Constants.textMessagePadding)
        .foregroundStyle(Constants.foregroundColor)
        .background(
          message.participant == .system
          ? Constants.systemMessageBackgroundColor
          : Constants.userMessageBackgroundColor
        )
        .clipShape(RoundedRectangle(cornerRadius: Constants.messageBackgroundCornerRadius))
      if message.participant == .system {
        Spacer()
      }
    }
    .listRowSeparator(.hidden)
  }
}

extension View {
  /// Displays error alert based on the value of the binding error. This function is invoked when the value of the binding error changes.
  /// - Parameters:
  ///   - error: Binding error based on which the alert is displayed.
  /// - Returns: The error alert.
  func alert(error: Binding<InferenceError?>, buttonTitle: String = "OK") -> some View {
    let inferenceError = error.wrappedValue
    return alert(isPresented: .constant(inferenceError != nil), error: inferenceError) { _ in
      Button(buttonTitle) {
        error.wrappedValue = nil
      }
    } message: { error in
      Text(error.failureReason)
    }
  }
}
