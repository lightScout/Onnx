//
//  TFLiteOnDeviceModel.swift
//  iosApp
//
//  Created by Juan Silva on 24/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

import Foundation
import TensorFlowLite

/// Error types specific to TFLite model operations
enum TFLiteError: Error {
    case modelNotFound
    case interpreterInitFailed
    case tensorAllocationFailed(String)
    case invalidInputData
    case inferenceError
    case internalError(Int32)
    
    var localizedDescription: String {
        switch self {
        case .modelNotFound:
            return "Model file not found"
        case .interpreterInitFailed:
            return "Failed to initialize TFLite interpreter"
        case .tensorAllocationFailed(let details):
            return "Failed to allocate tensors: \(details)"
        case .invalidInputData:
            return "Invalid input data format"
        case .inferenceError:
            return "Error during model inference"
        case .internalError(let code):
            return "Internal TFLite error: \(code)"
        }
    }
}

struct InferenceResult: Identifiable, Equatable {
    /// Represents the type of result
    enum Source {
        case input
        case output
    }
    
    /// Unique identifier for the result
    let id = UUID().uuidString
    /// Text content for input, logits array for output
    var content: ResultContent
    /// Indicates if this is input or output data
    let source: Source
    
    enum ResultContent: Equatable {
        case text(String)
        case logits([[Float]])
    }
}

/// Holds the names of the models that can be used
enum TFLiteModel: CaseIterable {
    case gpt2
    
    private var path: (name: String, extension: String) {
        switch self {
        case .gpt2:
            return ("GPT2-64-8bits", "tflite")
        }
    }
    
    var modelPath: String {
        get throws {
            guard let path = Bundle.main.path(
                forResource: path.name, ofType: path.extension)
            else {
                throw TFLiteInferenceError.modelFileNotFound(modelName: "\(path.name).\(path.extension)")
            }
            return path
        }
    }
}

struct TFLiteOnDeviceModel {
    /// TensorFlow Lite Interpreter
    private(set) var interpreter: Interpreter
    private(set) var session: TFLiteSession
    
    init(modelPath: String) throws {
        // Configure interpreter options
        var options = Interpreter.Options()
        options.threadCount = 2
        
        do {
            // Initialize interpreter
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            
            // Explicitly allocate tensors
            try interpreter.allocateTensors()
            
            // Verify tensor allocation
            guard interpreter.inputTensorCount > 0 else {
                throw TFLiteError.tensorAllocationFailed("No input tensors available")
            }
            
            // Print tensor details for debugging
            if let inputTensor = try? interpreter.input(at: 0) {
                print("Input tensor shape: \(inputTensor.shape.dimensions)")
                print("Input tensor type: \(inputTensor.dataType)")
            }
            
            session = try TFLiteSession(interpreter: interpreter)
        } catch let error as Error {
            print("TFLite initialization error: \(error.localizedDescription)")
            if let tfliteError = error as? TFLiteError {
                throw tfliteError
            } else {
                throw TFLiteError.internalError(4)
            }
        }
    }
}

class TFLiteSession {
    private let interpreter: Interpreter
    private let tokenizer: GPT2Tokenizer
    private var inputShape: [Int] = []
    private var outputShape: [Int] = []
    
    init(interpreter: Interpreter) throws {
        self.interpreter = interpreter
        self.tokenizer = GPT2Tokenizer()
        try setupShapes()
    }
    
    private func setupShapes() throws {
        do {
            guard let input = try? interpreter.input(at: 0) else {
                throw TFLiteError.tensorAllocationFailed("Could not access input tensor")
            }
            
            guard let output = try? interpreter.output(at: 0) else {
                throw TFLiteError.tensorAllocationFailed("Could not access output tensor")
            }
            
            inputShape = input.shape.dimensions
            outputShape = output.shape.dimensions
            
            print("Input shape: \(inputShape)")  // Should be [1, 64]
            print("Output shape: \(outputShape)") // Should be [1, 64, 50257]
        } catch {
            print("Setup shapes error: \(error.localizedDescription)")
            throw TFLiteError.tensorAllocationFailed("Failed to setup tensor shapes")
        }
    }
    
    func runInference(inputIds: [Int32]) throws -> [[Float]] {
        do {
            // Create input data with the correct shape [1, 64]
            let expectedInputSize = inputShape.reduce(1, *)
            var paddedInput = inputIds
            
            // Pad or truncate input to match expected size
            if paddedInput.count < expectedInputSize {
                paddedInput.append(contentsOf: Array(repeating: Int32(0), count: expectedInputSize - paddedInput.count))
            } else if paddedInput.count > expectedInputSize {
                paddedInput = Array(paddedInput.prefix(expectedInputSize))
            }
            
            // Convert to Data
            let inputData = Data(bytes: paddedInput, count: paddedInput.count * MemoryLayout<Int32>.size)
            
            // Copy input data
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output - reshaping to match [1, 64, 50257]
            guard let outputTensor = try? interpreter.output(at: 0),
                  let flatResults = outputTensor.data.toArray(type: Float32.self) else {
                throw TFLiteError.inferenceError
            }
            
            // Reshape flat array into expected dimensions
            let batchSize = outputShape[0]  // 1
            let sequenceLength = outputShape[1]  // 64
            let vocabularySize = outputShape[2]  // 50257
            
            var results: [[Float]] = []
            let stride = vocabularySize
            
            // Reshape the flat array into sequence_length arrays of vocabulary_size
            for i in 0..<sequenceLength {
                let start = i * stride
                let end = start + vocabularySize
                let slice = Array(flatResults[start..<end])
                results.append(slice)
            }
            
            return results
            
        } catch {
            print("Inference error: \(error.localizedDescription)")
            throw TFLiteError.inferenceError
        }
    }
    
    func processText(_ text: String) throws -> [[Float]] {
        // Tokenize input text
        var tokens = tokenizer.encode(text)
        
        // Ensure we have exactly 64 tokens (pad or truncate)
        let targetLength = inputShape[1] // Should be 64
//        if tokens.count < targetLength {
//            tokens.append(contentsOf: Array(repeating: tokenizer.padToken, count: targetLength - tokens.count))
//        } else if tokens.count > targetLength {
//            tokens = Array(tokens.prefix(targetLength))
//        }
        
        // Run inference
        return try runInference(inputIds: tokens)
    }
    
    func processTextAsync(_ text: String) -> AsyncThrowingStream<[[Float]], Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let results = try processText(text)
                    continuation.yield(results)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// Helper extension for converting Data to Array
extension Data {
    func toArray<T>(type: T.Type) -> [T]? where T: AdditiveArithmetic {
        var array = [T](repeating: T.zero, count: self.count / MemoryLayout<T>.size)
        _ = array.withUnsafeMutableBytes { self.copyBytes(to: $0) }
        return array
    }
}
