import TensorFlowLite
import UIKit

class TFLiteHandler {
    private var interpreter: Interpreter?
    private let modelPath: String
    
    init?(modelName: String) {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "tflite") else {
            print("Failed to load model")
            return nil
        }
        self.modelPath = modelPath
        setupInterpreter()
    }
    
    private func setupInterpreter() {
        do {
            // Configure interpreter options
            var options = Interpreter.Options()
            options.threadCount = 2
            
            // Create interpreter
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            
            // Allocate tensors
            try interpreter?.allocateTensors()
            
            print("Model input shape: \(try interpreter?.input(at: 0).shape ?? [])")
            print("Model output shape: \(try interpreter?.output(at: 0).shape ?? [])")
        } catch {
            print("Failed to create interpreter: \(error.localizedDescription)")
        }
    }
    
    func runInference(inputData: Data) -> [Float]? {
        do {
            // Copy input data to input tensor
            try interpreter?.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter?.invoke()
            
            // Get output tensor data
            let outputTensor = try interpreter?.output(at: 0)
            let results = outputTensor?.data.toArray(type: Float32.self)
            
            return results
        } catch {
            print("Failed to run inference: \(error.localizedDescription)")
            return nil
        }
    }
    
    // Helper method to preprocess image for model input
    func preprocessImage(_ image: UIImage, width: Int, height: Int) -> Data? {
        guard let cgImage = image.cgImage,
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return nil
        }
        
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let context = CGContext(data: nil,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: width * 4,
                              space: colorSpace,
                              bitmapInfo: bitmapInfo.rawValue)
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelBuffer = context?.data else {
            return nil
        }
        
        let data = Data(bytes: pixelBuffer, count: width * height * 4)
        return data
    }
}

// Example usage in a view controller
class ViewController: UIViewController {
    private var tfliteHandler: TFLiteHandler?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize TFLite handler with model
        tfliteHandler = TFLiteHandler(modelName: "model_name")
        
        // Example of running inference with an image
        if let image = UIImage(named: "test_image"),
           let inputData = tfliteHandler?.preprocessImage(image, width: 224, height: 224) {
            
            if let results = tfliteHandler?.runInference(inputData: inputData) {
                print("Inference results: \(results)")
            }
        }
    }
}

// Helper extension for converting Data to Array
extension Data {
    func toArray<T>(type: T.Type) -> [T] where T: AdditiveArithmetic {
        var array = [T](repeating: T.zero, count: self.count / MemoryLayout<T>.size)
        _ = array.withUnsafeMutableBytes { self.copyBytes(to: $0) }
        return array
    }
}
