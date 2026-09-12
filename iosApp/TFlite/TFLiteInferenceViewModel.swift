//
//  TFLiteInferenceViewModel.swift
//  iosApp
//
//  Created by Juan Silva on 24/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

import Foundation
import TensorFlowLite

/// Represents any error thrown by this application
enum TFLiteInferenceError: LocalizedError {
    /// Wraps an error thrown by TFLite
    case tensorFlowError(error: Error)
    case modelFileNotFound(modelName: String)
    case modelNotInitialized
    case invalidInputData
    
    public var errorDescription: String? {
        switch self {
        case .tensorFlowError:
            return "Internal error"
        case .modelFileNotFound:
            return "Model not found"
        case .modelNotInitialized:
            return "Model Uninitialized"
        case .invalidInputData:
            return "Invalid input data"
        }
    }
    
    public var failureReason: String {
        switch self {
        case .tensorFlowError(let error):
            return error.localizedDescription
        case .modelFileNotFound(let modelName):
            return "Model with name \(modelName) not found in your app."
        case .modelNotInitialized:
            return "A valid TFLite model has not been initialized."
        case .invalidInputData:
            return "The provided input data does not match the model's requirements."
        }
    }
}



@MainActor
class TFLiteInferenceViewModel: ObservableObject {
    /// This array holds both the input text and output logits
    @Published var results = [InferenceResult]()
    
    /// Indicates if we're waiting for the model to be initialized or finish inference
    @Published var busy = true
    
    /// Holds any error that occurred during inference
    @Published var error: TFLiteInferenceError?
    
    /// Model used for inference
    private var model: TFLiteOnDeviceModel?
    
    /// Current inference session
    private var inferenceSession: TFLiteSession?
    
    init() {
        defer {
            busy = false
        }
        do {
            let model = try TFLiteOnDeviceModel(modelPath: try TFLiteModel.gpt2.modelPath)
            self.model = model
            inferenceSession = model.session
        } catch let error as TFLiteInferenceError {
            self.error = error
        } catch {
            self.error = TFLiteInferenceError.tensorFlowError(error: error)
        }
    }
    
    /// Processes text input through the GPT-2 model asynchronously.
    /// - Parameters:
    ///   - text: Text to be processed by the model
    func processInput(_ text: String) {
        busy = true
        Task {
            await internalProcessInput(text)
            busy = false
        }
    }
    
    /// Clears the current session and starts a new one
    func startNewSession() {
        busy = true
        defer {
            busy = false
        }
        
        guard let model else {
            error = TFLiteInferenceError.modelNotInitialized
            return
        }
        
        inferenceSession = model.session
        results.removeAll()
    }
    
    /// Internal method to process text through the model
    private func internalProcessInput(_ text: String) async {
        guard let inferenceSession else {
            error = TFLiteInferenceError.modelNotInitialized
            return
        }
        
        // Add the input text to results
        let inputResult = InferenceResult(
            content: .text(text),
            source: .input
        )
        results.append(inputResult)
        
        do {
            // Get inference stream
            let resultStream = inferenceSession.processTextAsync(text)
            
            // Process results
            for try await logits in resultStream {
                guard let lastResult = self.results.last else { break }
                
                if lastResult.source == .input {
                    // Add new output result
                    results.append(InferenceResult(
                        content: .logits(logits),
                        source: .output
                    ))
                } else {
                    // Update existing output result
                    var updatedResults = self.results
                    updatedResults[updatedResults.count - 1] = InferenceResult(
                        content: .logits(logits),
                        source: .output
                    )
                    self.results = updatedResults
                }
            }
        } catch {
            self.error = TFLiteInferenceError.tensorFlowError(error: error)
            results.removeLast()
        }
    }
    
    /// Gets the most likely next token for the given logits
    func getNextToken(from logits: [[Float]]) -> String? {
        guard let inferenceSession = inferenceSession else { return nil }
        
        // For now, just get the highest probability token from the last position
        if let lastPositionLogits = logits.last {
            if let maxIndex = lastPositionLogits.indices.max(by: { lastPositionLogits[$0] < lastPositionLogits[$1] }) {
                // Convert token ID back to text
                let tokenId = Int32(maxIndex)
                // You would need to add a method to your tokenizer to decode single tokens
                return "\(tokenId)" // For now just returning the ID
            }
        }
        return nil
    }
    
    /// Format logits for display
    func formatLogits(_ logits: [[Float]]) -> String {
        let maxTokens = 5 // Show top 5 probabilities for the last position
        
        guard let lastPositionLogits = logits.last else {
            return "No logits available"
        }
        
        // Get indices sorted by probability
        let sortedIndices = lastPositionLogits.indices.sorted {
            lastPositionLogits[$0] > lastPositionLogits[$1]
        }
        
        // Format top probabilities
        let topLogits = sortedIndices.prefix(maxTokens).map { index in
            let probability = lastPositionLogits[index]
            return "Token \(index): \(String(format: "%.4f", probability))"
        }
        
        return topLogits.joined(separator: "\n")
    }
}

