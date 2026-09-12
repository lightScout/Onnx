// Copyright 2024 The MediaPipe Authors.
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

import Foundation
import MediaPipeTasksGenAI

/// Represents a single message in the chat.
struct ChatMessage: Identifiable, Equatable {

  /// Represents the type of message.
  enum Participant {
    case system
    case user
  }

  /// Unique identifier for the message.
  let id = UUID().uuidString
  /// Text contained in the message.
  var text: String
  /// Indicates if user or system (LLM) has sent the message.
  let participant: Participant

  init(text: String = "", participant: Participant) {
    self.text = text
    self.participant = participant
  }

}

/// Represents any error thrown by this application.
enum InferenceError: LocalizedError {
    case mediaPipeTasksError(error: Error)
    case modelFileNotFound(modelName: String)
    case onDeviceModelNotInitialized
    case modelDownloadFailed(error: Error)
    

    public var errorDescription: String? {
        switch self {
        case .mediaPipeTasksError:
             return "Internal error"
           case .modelFileNotFound:
             return "Model not found"
           case .onDeviceModelNotInitialized:
             return "Model Unitialized"
        case .modelDownloadFailed:
            return "Model Download Failed"
        }
    }

    public var failureReason: String {
        switch self {
        case .mediaPipeTasksError(let error):
          return error.localizedDescription
        case .modelFileNotFound(let modelName):
          return "Model with name \(modelName) not found in your app."
        case .onDeviceModelNotInitialized:
          return "A valid on device model has not been initalized."
        case .modelDownloadFailed(let error):
            return error.localizedDescription
        }
    }
}

/// Holds the names of the models that can be used.
enum Model: CaseIterable {
    case gemma

    private var pathComponents: (name: String, `extension`: String) {
        switch self {
        case .gemma:
            return ("Llama-3.2-1b-q8", "task")
        }
    }

    /// Returns the model file URL in the documents directory.
    var modelFileURL: URL {
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsURL.appendingPathComponent("\(pathComponents.name).\(pathComponents.extension)")
    }

    /// Checks if the model file exists in the documents directory.
    var modelExists: Bool {
        let fileManager = FileManager.default
        return fileManager.fileExists(atPath: modelFileURL.path)
    }
}

/// Represents the download state of the model.
enum ModelDownloadState: Equatable {
    case notStarted
    case downloading(progress: Double)
    case downloaded
    case failed(error: Error)
    static func == (lhs: ModelDownloadState, rhs: ModelDownloadState) -> Bool {
           switch (lhs, rhs) {
           case (.notStarted, .notStarted):
               return true
           case (.downloading(let lhsProgress), .downloading(let rhsProgress)):
               return lhsProgress == rhsProgress
           case (.downloaded, .downloaded):
               return true
           case (.failed(_), .failed(_)):
               // Both are failed states; errors are not compared
               return true
           default:
               return false
           }
       }
}

@MainActor
class ConversationViewModel: NSObject, ObservableObject  {
    /// This array holds both the user's and the system's chat messages.
    @Published var messages = [ChatMessage]()

    /// Indicates if we're waiting for the model to be initialized or finish generating a response.
    @Published var busy = true

    /// Holds any error that occurs during inference.
    @Published var error: InferenceError?

    /// Tracks the model's download state.
    @Published var modelDownloadState: ModelDownloadState = .notStarted

    /// Model used for inference.
    private var model: OnDeviceModel?

    /// Current conversation with the LLM that preserves history.
    private var chat: Chat?

    /// Continuation used for the download completion.
    private var downloadCompletionContinuation: CheckedContinuation<Void, Error>?

    override init() {
        super.init()
        Task { [weak self] in
            await self?.initializeModel()
        }
    }

    /// Initializes the model by ensuring it's downloaded and ready to use.
    private func initializeModel() async {
        self.busy = true
        do {
            if !Model.gemma.modelExists {
                try await downloadModel()
            } else {
                DispatchQueue.main.async {
                    self.modelDownloadState = .downloaded
                }
            }

            let model = try OnDeviceModel(model: Model.gemma)
            self.model = model
            chat = try Chat(inference: model.session)
        } catch let error as InferenceError {
            self.error = error
        } catch {
            self.error = InferenceError.mediaPipeTasksError(error: error)
        }
        self.busy = false
    }

    /// Downloads the model file to the documents directory.
    private func downloadModel() async throws {
        guard let url = URL(string: "https://huggingface.co/vimal-yuvabe/llama-3.2-1b-tflite/tree/main/llama-3.2-1b-q8.task)") else {
            throw InferenceError.modelDownloadFailed(error: NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid model URL"]))
        }

        self.modelDownloadState = .downloading(progress: 0.0)

        let sessionConfig = URLSessionConfiguration.default
        let session = URLSession(configuration: sessionConfig, delegate: self, delegateQueue: nil)

        try await withCheckedThrowingContinuation { continuation in
            self.downloadCompletionContinuation = continuation
            let downloadTask = session.downloadTask(with: url)
            downloadTask.resume()
        }
    }

    /// Queries the LLM session with the given text prompt asynchronously.
    func sendMessage(_ text: String) {
        busy = true
        Task {
            await internalSendMessage(text)
            busy = false
        }
    }

    /// Clears the current conversation and starts a new chat.
    func startNewChat() {
        busy = true
        defer {
            busy = false
        }

        guard let model else {
            error = InferenceError.onDeviceModelNotInitialized
            return
        }
        do {
            chat = try Chat(inference: model.session)
            messages.removeAll()
        } catch {
            self.error = InferenceError.mediaPipeTasksError(error: error)
        }
    }

    /// Sends the message to the currently active instance of `Chat`.
    private func internalSendMessage(_ text: String) async {
        guard let chat else {
            error = InferenceError.onDeviceModelNotInitialized
            return
        }
        // Add the user's message to the chat.
        let userMessage = ChatMessage(text: text, participant: .user)
        messages.append(userMessage)

        do {
            // Send the message to the chat session to query the model.
            let responseStream = try await chat.sendMessage(text)

            // Await for the partial responses from the model.
            for try await partialResult in responseStream {
                // Ensure there is a last message.
                guard let lastMessage = self.messages.last else {
                    break
                }

                // If this is the first partial response, add a new LLM message to the chat.
                if lastMessage.participant == .user {
                    messages.append(ChatMessage(participant: .system))
                }
                let systemMessageText = messages[messages.count - 1].text

                // Trim any leading characters in whole message.
                messages[messages.count - 1].text = String(
                    (systemMessageText + partialResult).drop(while: { $0.isWhitespace || $0.isNewline }))
            }
        } catch {
            // Handle errors thrown during message processing.
            self.error = InferenceError.mediaPipeTasksError(error: error)
            messages.removeLast()
        }
    }
}

extension ConversationViewModel: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        if totalBytesExpectedToWrite > 0 {
            let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            DispatchQueue.main.async {
                self.modelDownloadState = .downloading(progress: progress)
            }
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let fileManager = FileManager.default
        let modelURL = Model.gemma.modelFileURL
        do {
            // Remove existing file if it exists
            if fileManager.fileExists(atPath: modelURL.path) {
                try fileManager.removeItem(at: modelURL)
            }
            // Move downloaded file to the documents directory
            try fileManager.moveItem(at: location, to: modelURL)
            DispatchQueue.main.async {
                self.modelDownloadState = .downloaded
            }
            self.downloadCompletionContinuation?.resume()
            self.downloadCompletionContinuation = nil
        } catch {
            DispatchQueue.main.async {
                self.modelDownloadState = .failed(error: error)
            }
            self.downloadCompletionContinuation?.resume(throwing: error)
            self.downloadCompletionContinuation = nil
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error = error {
            DispatchQueue.main.async {
                self.modelDownloadState = .failed(error: error)
            }
            self.downloadCompletionContinuation?.resume(throwing: error)
            self.downloadCompletionContinuation = nil
        }
    }
}
