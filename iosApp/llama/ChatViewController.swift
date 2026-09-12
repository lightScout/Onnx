//
//  ChatViewController.swift
//  iosApp
//
//  Created by Juan Silva on 23/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

//import UIKit
//import LibTorch
//
//class ChatViewController: UIViewController {
//
//    @IBOutlet weak var inputTextField: UITextField!
//    @IBOutlet weak var chatTableView: UITableView!
//    @IBOutlet weak var sendButton: UIButton!
//
//    var module: TorchModule?
//    var conversationHistory: [String] = []
//
//    override func viewDidLoad() {
//        super.viewDidLoad()
//
//        // Load the TorchScript model
//        if let filePath = Bundle.main.path(forResource: "qwen2.5_0.5b_instruct_optimized", ofType: "pt"),
//           let module = TorchModule(fileAtPath: filePath) {
//            self.module = module
//        } else {
//            fatalError("Failed to load model.")
//        }
//
//        // Set up the chat table view
//        chatTableView.dataSource = self
//        chatTableView.delegate = self
//
//        // Dismiss keyboard on tap
//        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
//        chatTableView.addGestureRecognizer(tapGesture)
//    }
//
//    @objc func dismissKeyboard() {
//        inputTextField.resignFirstResponder()
//    }
//
//    @IBAction func sendButtonTapped(_ sender: UIButton) {
//        guard let inputText = inputTextField.text, !inputText.isEmpty else { return }
//
//        // Add user's message to the conversation history
//        conversationHistory.append("User: \(inputText)")
//        chatTableView.reloadData()
//
//        // Clear the input field
//        inputTextField.text = ""
//
//        // Generate response from the model
//        DispatchQueue.global().async {
//            if let response = self.generateResponse(inputText: inputText) {
//                DispatchQueue.main.async {
//                    self.conversationHistory.append("Bot: \(response)")
//                    self.chatTableView.reloadData()
//
//                    // Scroll to the bottom
//                    let indexPath = IndexPath(row: self.conversationHistory.count - 1, section: 0)
//                    self.chatTableView.scrollToRow(at: indexPath, at: .bottom, animated: true)
//                }
//            }
//        }
//    }
//
//    func generateResponse(inputText: String) -> String? {
//        guard let module = self.module else { return nil }
//
//        // Preprocess input text
//        guard let inputTensor = preprocessInput(inputText) else { return nil }
//
//        // Run the model
//        guard let outputTensor = module.forward(inputTensor) else { return nil }
//
//        // Postprocess output tensor to get response text
//        let responseText = postprocessOutput(outputTensor)
//        return responseText
//    }
//
//    func preprocessInput(_ text: String) -> TorchTensor? {
//        // Simple character-level encoding (for demonstration)
//        // Convert each character to its ASCII value
//        let asciiValues = text.compactMap { character -> Int32? in
//            guard let ascii = character.asciiValue else { return nil }
//            return Int32(ascii)
//        }
//
//        // Create a tensor from the ASCII values
//        let tensorData = Data(buffer: UnsafeBufferPointer(start: asciiValues, count: asciiValues.count))
//        let tensor = TorchTensor(data: tensorData, shape: [1, NSNumber(value: asciiValues.count)], dtype: .int32)
//        return tensor
//    }
//
//    func postprocessOutput(_ outputTensor: TorchTensor) -> String {
//        // Convert output tensor to string
//        let data = outputTensor.data
//        let count = outputTensor.shape.reduce(1) { x, y in x.intValue * y.intValue }
//        let pointer = data.bindMemory(to: Int32.self, capacity: count)
//        var outputText = ""
//        for i in 0..<count {
//            let asciiValue = Int(pointer[i])
//            if let scalar = UnicodeScalar(asciiValue) {
//                outputText.append(Character(scalar))
//            }
//        }
//        return outputText
//    }
//}
//
//// MARK: - UITableViewDataSource, UITableViewDelegate
//
//extension ChatViewController: UITableViewDataSource, UITableViewDelegate {
//    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
//       return conversationHistory.count
//    }
//
//    func numberOfSections(in tableView: UITableView) -> Int {
//       return 1
//    }
//
//    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
//
//        let cellIdentifier = "ChatCell"
//
//        let cell = tableView.dequeueReusableCell(withIdentifier: cellIdentifier) ??
//                   UITableViewCell(style: .default, reuseIdentifier: cellIdentifier)
//
//        cell.textLabel?.text = conversationHistory[indexPath.row]
//        cell.textLabel?.numberOfLines = 0  // Allow multiple lines
//        return cell
//    }
//}

