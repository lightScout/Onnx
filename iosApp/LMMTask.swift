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
import MediaPipeTasksGenAIC

/// Represents the LLM that will be used for inference.  It manages a MediaPipe `LlmInference` under the hood.
struct OnDeviceModel {
  /// MediaPipe LlmInference.
  private(set) var inference: LlmInference

  init(model: Model) throws {
    inference = try LlmInference(modelPath: try model.modelPath)
  }
}

/// Represents a chat session using an instance of `OnDeviceModel`.  It manages a MediaPipe
/// `LlmInference.Session` under the hood and passes all response generation queries to the session.
final class Chat {
  /// The on device inference engine for this chat session
  private let inference: LlmInference

  init(inference: LlmInference) throws {
    self.inference = inference
  }
  
  /// Sends a streaming response generation query to the underlying MediaPipe
  /// `LlmInference.Session`.
  /// - Parameters:
  ///   - text: Query to the underlying LLM.
  /// - Returns: An async throwing stream that contains the partial responses from the LLM.
  /// - Throws: A MediaPipe `GenAiInferenceError` if the query cannot be added to the current
  /// session.
  func sendMessage(_ text: String) async throws -> AsyncThrowingStream<String, any Error> {
    let resultStream = inference.generateResponseAsync(inputText: text)
    return resultStream
  }
}

/// Holds the names of the models names that can be  used.
enum Model: CaseIterable {
  case gemma

  private var path: (name: String, extension: String) {
    switch self {
    case .gemma:
      return ("gemma-2b-it-cpu-int4", "bin")
    }
  }

  var modelPath: String {
    get throws {
      guard
        let path = Bundle.main.path(
          forResource: path.name, ofType: path.extension)
      else {
        throw InferenceError.modelFileNotFound(modelName: "\(path.name).\(path.extension)")
      }
      return path
    }
  }
}

/// Represents any error thrown by this application.
enum InferenceError: LocalizedError {
  /// Wraps an error thrown by MediaPipe.
  case mediaPipeTasksError(error: Error)
  case modelFileNotFound(modelName: String)
  case onDeviceModelNotInitialized

  public var errorDescription: String? {
    switch self {
    case .mediaPipeTasksError:
      return "Internal error"
    case .modelFileNotFound:
      return "Model not found"
    case .onDeviceModelNotInitialized:
      return "Model Unitialized"
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
    }
  }

}
