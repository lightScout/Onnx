//
//  OnDeviceModel.swift
//  iosApp
//
//  Created by Juan Silva on 28/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

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
struct GemmaOnDeviceModel {
  /// MediaPipe LlmInference.
  private(set) var inference: LlmInference
  private(set) var session: LlmInference.Session

  init(model: GemmaModel) throws {
      let options = try LlmInference.Options(modelPath: model.modelPath)
          options.maxTokens = 1200
          inference = try LlmInference(options: options)
          let sessionOptions = LlmInference.Session.Options()
          sessionOptions.temperature = 0.2
//          sessionOptions.randomSeed = 2222
          session = try LlmInference.Session(llmInference: inference, options: sessionOptions)
    
  }
}

/// Represents a chat session using an instance of `OnDeviceModel`.  It manages a MediaPipe
/// `LlmInference.Session` under the hood and passes all response generation queries to the session.
final class GemmaChat {
  /// The on device inference engine for this chat session
    private let inference: LlmInference.Session

    init(inference: LlmInference.Session) throws {
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
        try inference.addQueryChunk(inputText: text)
        let resultStream = inference.generateResponseAsync()
      return resultStream
    }
}
