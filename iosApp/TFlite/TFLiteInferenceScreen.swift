import SwiftUI

struct TFLiteInferenceScreen: View {
    private struct Constants {
        static let scrollDelayInSeconds = 0.05
        static let inputFieldPlaceholder = "Enter text..."
        static let newSessionSymbolName = "arrow.clockwise"
        static let navigationTitle = "GPT-2 Inference"
        static let processButtonText = "Generate"
    }
    
    @EnvironmentObject
    var viewModel: TFLiteInferenceViewModel
    
    @State
    private var currentInput = ""
    
    private enum FocusedField: Hashable {
        case input
    }
    
    @FocusState
    private var focusedField: FocusedField?
    
    var body: some View {
        NavigationView {
            VStack {
                ScrollViewReader { scrollViewProxy in
                    if #available(iOS 17.0, *) {
                        List {
                            ForEach(viewModel.results) { result in
                                ResultView(result: result, viewModel: viewModel)
                            }
                        }
                        .listStyle(.plain)
                        .onChange(of: viewModel.results) { _, newValue in
                            Task { @MainActor in
                                guard let lastResult = viewModel.results.last else { return }
                                try await Task.sleep(for: .seconds(Constants.scrollDelayInSeconds))
                                withAnimation {
                                    scrollViewProxy.scrollTo(lastResult.id, anchor: .bottom)
                                }
                                focusedField = .input
                            }
                        }
                    } else {
                        // Fallback on earlier versions
                    }
                }
                
                HStack {
                    TextField(Constants.inputFieldPlaceholder, text: $currentInput)
                        .focused($focusedField, equals: .input)
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                        .disabled(viewModel.busy)
                        .padding()
                    
                    Button(Constants.processButtonText) {
                        processInput()
                    }
                    .disabled(viewModel.busy || currentInput.isEmpty)
                    .padding(.trailing)
                }
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: viewModel.startNewSession) {
                        Image(systemName: Constants.newSessionSymbolName)
                    }
                    .disabled(viewModel.busy)
                }
            }
            .alert(error: $viewModel.error)
            .navigationTitle(Constants.navigationTitle)
            .onAppear {
                focusedField = .input
            }
        }
    }
    
    private func processInput() {
        guard !currentInput.isEmpty else { return }
        let input = currentInput
        currentInput = ""
        viewModel.processInput(input)
    }
}

/// View that displays an inference result
struct ResultView: View {
    private struct Constants {
        static let resultPadding: CGFloat = 10.0
        static let foregroundColor = Color(red: 0.0, green: 0.0, blue: 0.0)
        static let outputResultBackgroundColor = Color(white: 0.9231)
        static let inputResultBackgroundColor = Color(red: 0.8627, green: 0.9725, blue: 0.7764)
        static let resultBackgroundCornerRadius: CGFloat = 16.0
    }
    
    /// Result to be displayed
    var result: InferenceResult
    let viewModel: TFLiteInferenceViewModel
    
    var body: some View {
        HStack {
            if result.source == .input {
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(result.source == .input ? "Input:" : "Output:")
                    .font(.caption)
                    .foregroundColor(.gray)
                
                Text(formatContent(result.content))
                    .padding(Constants.resultPadding)
                    .foregroundStyle(Constants.foregroundColor)
            }
            .background(
                result.source == .output
                ? Constants.outputResultBackgroundColor
                : Constants.inputResultBackgroundColor
            )
            .clipShape(RoundedRectangle(cornerRadius: Constants.resultBackgroundCornerRadius))
            
            if result.source == .output {
                Spacer()
            }
        }
        .listRowSeparator(.hidden)
    }
    
    private func formatContent(_ content: InferenceResult.ResultContent) -> String {
        switch content {
        case .text(let text):
            return text
        case .logits(let logits):
            return viewModel.formatLogits(logits)
        }
    }
}

extension View {
    /// Displays error alert based on the value of the binding error
    /// - Parameters:
    ///   - error: Binding error based on which the alert is displayed
    /// - Returns: The error alert
    func alert(error: Binding<TFLiteInferenceError?>, buttonTitle: String = "OK") -> some View {
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
