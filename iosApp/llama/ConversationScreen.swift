// Copyright 2024 The Mediapipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftUI

struct ConversationScreen: View {
    private struct Constants {
        static let scrollDelayInSeconds = 0.05
        static let messageFieldPlaceHolder = "Message..."
        static let newChatSystemSymbolName = "square.and.pencil"
        static let navigationTitle = "Chat with your LLM here"
    }

    @EnvironmentObject var viewModel: ConversationViewModel

    @State private var currentUserPrompt = ""

    private enum FocusedField: Hashable {
        case message
    }

    @FocusState private var focusedField: FocusedField?

    var body: some View {
        NavigationView {
            VStack {
                switch viewModel.modelDownloadState {
                case .downloading(let progress):
                    ProgressView(value: progress)
                        .padding()
                    Text("Downloading model... \(Int(progress * 100))%")
                        .padding()
                case .failed(let error):
                    Text("Failed to download model: \(error.localizedDescription)")
                        .foregroundColor(.red)
                        .padding()
                case .notStarted:
                    Text("Initializing...")
                        .padding()
                case .downloaded:
                    contentView
                }
            }
            .navigationTitle(Constants.navigationTitle)
            .onAppear {
                focusedField = .message
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: viewModel.startNewChat) {
                        Image(systemName: Constants.newChatSystemSymbolName)
                    }.disabled(viewModel.busy || viewModel.modelDownloadState != .downloaded)
                }
            }
            .alert(error: $viewModel.error)
        }
    }

    @ViewBuilder
    private var contentView: some View {
        ScrollViewReader { scrollViewProxy in
            if #available(iOS 17.0, *) {
                List {
                    ForEach(viewModel.messages) { message in
                        MessageView(message: message)
                    }
                }
                .listStyle(.plain)
                .onChange(of: viewModel.messages) { _, _ in
                    Task { @MainActor in
                        guard let lastMessage = viewModel.messages.last else { return }
                        try await Task.sleep(for: .seconds(Constants.scrollDelayInSeconds))
                        withAnimation {
                            scrollViewProxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                        focusedField = .message
                    }
                }
            } else {
                // Fallback on earlier versions
            }
            TextField(Constants.messageFieldPlaceHolder, text: $currentUserPrompt)
                .focused($focusedField, equals: .message)
                .onSubmit {
                    sendMessage()
                }
                .submitLabel(.send)
                .disabled(viewModel.busy || viewModel.modelDownloadState != .downloaded)
                .padding()
        }
    }

    private func sendMessage() {
        guard !currentUserPrompt.isEmpty else {
            return
        }
        let prompt = currentUserPrompt
        currentUserPrompt = ""
        viewModel.sendMessage(prompt)
    }
}

/// View that displays a message.
struct MessageView: View {
    private struct Constants {
        static let textMessagePadding: CGFloat = 10.0
        static let foregroundColor = Color(red: 0.0, green: 0.0, blue: 0.0)
        static let systemMessageBackgroundColor = Color(white: 0.9231)
        static let userMessageBackgroundColor = Color(red: 0.8627, green: 0.9725, blue: 0.7764)
        static let messageBackgroundCornerRadius: CGFloat = 16.0
    }
    /// Message to be displayed.
    var message: ChatMessage

    var body: some View {
        HStack {
            if message.participant == .user {
                Spacer()
            }
            Text(message.text)
                .padding(Constants.textMessagePadding)
                .foregroundColor(Constants.foregroundColor)
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


