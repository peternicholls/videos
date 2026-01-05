/// CoreMLExport.swift
/// Export MLSwift models to CoreML-compatible format for iOS/macOS deployment
/// Pure Swift implementation

import Foundation

// MARK: - CoreML Export

/// CoreML model export utilities
public enum CoreMLExport {
    
    /// Export a sequential model to a CoreML-compatible JSON specification
    /// This JSON format can be loaded back into MLSwift or used as a reference
    /// for building CoreML models using CreateML or the CoreML framework
    /// - Parameters:
    ///   - model: The MLSwift model to export
    ///   - inputName: Name for the input tensor
    ///   - inputShape: Shape of the input (e.g., [1, 784] for MNIST)
    ///   - outputName: Name for the output tensor
    /// - Returns: JSON string containing the model specification
    public static func exportToJSON(
        model: SequentialModel,
        inputName: String = "input",
        inputShape: [Int],
        outputName: String = "output"
    ) throws -> String {
        var spec: [String: Any] = [:]
        
        // Model metadata
        spec["format"] = "MLSwift-CoreML-Export"
        spec["version"] = "1.0"
        spec["inputName"] = inputName
        spec["inputShape"] = inputShape
        spec["outputName"] = outputName
        
        // Export layers
        var layerSpecs: [[String: Any]] = []
        
        for (index, layer) in model.getLayers().enumerated() {
            if let layerSpec = try exportLayer(layer, index: index) {
                layerSpecs.append(layerSpec)
            }
        }
        
        spec["layers"] = layerSpecs
        
        // Convert to JSON
        let jsonData = try JSONSerialization.data(withJSONObject: spec, options: [.prettyPrinted, .sortedKeys])
        guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            throw CoreMLExportError.jsonEncodingFailed
        }
        
        return jsonString
    }
    
    /// Export a single layer to a dictionary specification
    private static func exportLayer(_ layer: Layer, index: Int) throws -> [String: Any]? {
        var spec: [String: Any] = [:]
        spec["index"] = index
        
        switch layer {
        case let dense as DenseLayer:
            spec["type"] = "innerProduct"
            spec["name"] = "dense_\(index)"
            spec["inputChannels"] = dense.weights.cols
            spec["outputChannels"] = dense.weights.rows
            spec["hasBias"] = true
            
            // Export weights in row-major format
            spec["weights"] = dense.weights.data
            spec["bias"] = dense.bias.data
            
        case _ as ReLULayer:
            spec["type"] = "activation"
            spec["name"] = "relu_\(index)"
            spec["activationType"] = "ReLU"
            
        case _ as SigmoidLayer:
            spec["type"] = "activation"
            spec["name"] = "sigmoid_\(index)"
            spec["activationType"] = "sigmoid"
            
        case _ as TanhLayer:
            spec["type"] = "activation"
            spec["name"] = "tanh_\(index)"
            spec["activationType"] = "tanh"
            
        case _ as SoftmaxLayer:
            spec["type"] = "softmax"
            spec["name"] = "softmax_\(index)"
            
        case let batchNorm as BatchNormLayer:
            spec["type"] = "batchnorm"
            spec["name"] = "batchnorm_\(index)"
            spec["channels"] = batchNorm.gamma.rows
            spec["gamma"] = batchNorm.gamma.data
            spec["beta"] = batchNorm.beta.data
            spec["mean"] = batchNorm.runningMean.data
            spec["variance"] = batchNorm.runningVar.data
            
        case let dropout as DropoutLayer:
            // Dropout is typically not included in inference models
            // But we record it for completeness
            spec["type"] = "dropout"
            spec["name"] = "dropout_\(index)"
            spec["rate"] = dropout.rate
            spec["note"] = "Dropout is disabled during inference"
            
        case let conv2d as Conv2DLayer:
            spec["type"] = "convolution"
            spec["name"] = "conv2d_\(index)"
            spec["inputChannels"] = conv2d.inputChannels
            spec["outputChannels"] = conv2d.outputChannels
            spec["kernelSize"] = conv2d.kernelSize
            spec["stride"] = conv2d.stride
            spec["weights"] = conv2d.weights.data
            spec["bias"] = conv2d.bias.data
            
        case let maxPool as MaxPool2DLayer:
            spec["type"] = "pooling"
            spec["name"] = "maxpool_\(index)"
            spec["poolType"] = "max"
            spec["kernelSize"] = maxPool.poolSize
            spec["stride"] = maxPool.stride
            
        case _ as FlattenLayer:
            spec["type"] = "flatten"
            spec["name"] = "flatten_\(index)"
            
        case let lstm as LSTMLayer:
            spec["type"] = "lstm"
            spec["name"] = "lstm_\(index)"
            spec["inputSize"] = lstm.inputSize
            spec["hiddenSize"] = lstm.hiddenSize
            spec["returnSequences"] = lstm.returnSequences
            spec["weightsIH"] = lstm.weightsIH.data
            spec["weightsHH"] = lstm.weightsHH.data
            spec["bias"] = lstm.bias.data
            
        case let gru as GRULayer:
            spec["type"] = "gru"
            spec["name"] = "gru_\(index)"
            spec["inputSize"] = gru.inputSize
            spec["hiddenSize"] = gru.hiddenSize
            spec["returnSequences"] = gru.returnSequences
            spec["weightsIH"] = gru.weightsIH.data
            spec["weightsHH"] = gru.weightsHH.data
            spec["bias"] = gru.bias.data
            
        case let embedding as EmbeddingLayer:
            spec["type"] = "embedding"
            spec["name"] = "embedding_\(index)"
            spec["vocabSize"] = embedding.vocabSize
            spec["embeddingDim"] = embedding.embeddingDim
            spec["weights"] = embedding.embeddings.data
            
        default:
            // Skip unknown layer types
            return nil
        }
        
        return spec
    }
    
    /// Save the model specification to a JSON file
    /// - Parameters:
    ///   - model: The model to export
    ///   - url: Destination URL for JSON file
    ///   - inputName: Name for the input tensor
    ///   - inputShape: Shape of the input tensor
    ///   - outputName: Name for the output tensor
    public static func save(
        model: SequentialModel,
        to url: URL,
        inputName: String = "input",
        inputShape: [Int],
        outputName: String = "output"
    ) throws {
        let json = try exportToJSON(
            model: model,
            inputName: inputName,
            inputShape: inputShape,
            outputName: outputName
        )
        try json.write(to: url, atomically: true, encoding: .utf8)
    }
    
    /// Load a model specification from a JSON file
    /// - Parameter url: URL of the JSON specification file
    /// - Returns: Dictionary containing the model specification
    public static func loadSpecification(from url: URL) throws -> [String: Any] {
        let jsonData = try Data(contentsOf: url)
        guard let spec = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw CoreMLExportError.invalidModelStructure
        }
        return spec
    }
    
    /// Export model specification as a Swift code snippet that can create
    /// the equivalent CoreML model using the CoreML framework APIs
    /// - Parameters:
    ///   - model: The MLSwift model to export
    ///   - inputName: Name for the input tensor
    ///   - inputShape: Shape of the input tensor
    ///   - outputName: Name for the output tensor
    /// - Returns: Swift code as a string
    public static func generateSwiftCoreMLCode(
        model: SequentialModel,
        inputName: String = "input",
        inputShape: [Int],
        outputName: String = "output"
    ) -> String {
        var code = """
        // Generated CoreML model builder code
        // Copy this into your macOS/iOS project to create a CoreML model
        
        import CoreML
        
        @available(macOS 10.13, iOS 11.0, *)
        func createModel() throws -> MLModel {
            // Note: For production use, consider using CreateML or 
            // training your model directly with Core ML APIs
            
            // Model architecture from MLSwift:
            // Input: \(inputName) with shape \(inputShape)
            // Output: \(outputName)
            //
            // Layers:
        
        """
        
        for (index, layer) in model.getLayers().enumerated() {
            code += "    // Layer \(index): \(String(describing: type(of: layer)))\n"
            
            switch layer {
            case let dense as DenseLayer:
                code += "    //   Dense: \(dense.weights.cols) -> \(dense.weights.rows)\n"
            case _ as ReLULayer:
                code += "    //   Activation: ReLU\n"
            case _ as SigmoidLayer:
                code += "    //   Activation: Sigmoid\n"
            case _ as TanhLayer:
                code += "    //   Activation: Tanh\n"
            case _ as SoftmaxLayer:
                code += "    //   Activation: Softmax\n"
            case let batchNorm as BatchNormLayer:
                code += "    //   BatchNorm: \(batchNorm.gamma.rows) channels\n"
            case _ as DropoutLayer:
                code += "    //   Dropout (skipped for inference)\n"
            case let conv2d as Conv2DLayer:
                code += "    //   Conv2D: \(conv2d.inputChannels) -> \(conv2d.outputChannels), kernel \(conv2d.kernelSize)\n"
            case let maxPool as MaxPool2DLayer:
                code += "    //   MaxPool: \(maxPool.poolSize)x\(maxPool.poolSize)\n"
            case _ as FlattenLayer:
                code += "    //   Flatten\n"
            case let lstm as LSTMLayer:
                code += "    //   LSTM: input \(lstm.inputSize), hidden \(lstm.hiddenSize)\n"
            case let gru as GRULayer:
                code += "    //   GRU: input \(gru.inputSize), hidden \(gru.hiddenSize)\n"
            case let embedding as EmbeddingLayer:
                code += "    //   Embedding: vocab \(embedding.vocabSize), dim \(embedding.embeddingDim)\n"
            default:
                code += "    //   Unknown layer type\n"
            }
        }
        
        code += """
            
            // To create a CoreML model programmatically, use:
            // 1. MLModelDescription and MLModel for simple inference
            // 2. CreateML for training and model creation
            // 3. The exported JSON weights from MLSwift
            
            // Load the exported JSON to get the weights:
            // let spec = try CoreMLExport.loadSpecification(from: jsonURL)
            
            fatalError("Implement model creation using CoreML or CreateML APIs")
        }
        """
        
        return code
    }
}

// MARK: - CoreML Export Errors

public enum CoreMLExportError: Error, LocalizedError {
    case unsupportedLayerType(String)
    case jsonEncodingFailed
    case invalidModelStructure
    case coreMLNotAvailable
    case exportFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .unsupportedLayerType(let type):
            return "Unsupported layer type for CoreML export: \(type)"
        case .jsonEncodingFailed:
            return "Failed to encode model to JSON"
        case .invalidModelStructure:
            return "Model structure is invalid for CoreML export"
        case .coreMLNotAvailable:
            return "CoreML is not available on this platform"
        case .exportFailed(let reason):
            return "CoreML export failed: \(reason)"
        }
    }
}

// MARK: - Model Extension for Export

extension SequentialModel {
    
    /// Export model to JSON specification format
    /// - Parameters:
    ///   - url: Destination URL (should end in .json)
    ///   - inputShape: Shape of the input tensor
    /// - Throws: CoreMLExportError if export fails
    public func exportForCoreML(to url: URL, inputShape: [Int]) throws {
        try CoreMLExport.save(
            model: self,
            to: url,
            inputShape: inputShape
        )
    }
    
    /// Generate Swift code that documents the model architecture
    /// for creating an equivalent CoreML model
    /// - Parameters:
    ///   - inputShape: Shape of the input tensor
    /// - Returns: Swift code as a string
    public func generateCoreMLSwiftCode(inputShape: [Int]) -> String {
        return CoreMLExport.generateSwiftCoreMLCode(
            model: self,
            inputShape: inputShape
        )
    }
}

// MARK: - Usage Example

/*
 Usage (Pure Swift - No Python Required):
 
 1. Export your trained MLSwift model to JSON:
 
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 784, outputSize: 128))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 128, outputSize: 10))
    model.add(SoftmaxLayer())
    
    // Train the model...
    
    // Export to JSON specification (contains all weights and architecture)
    try model.exportForCoreML(
        to: URL(fileURLWithPath: "model_spec.json"),
        inputShape: [1, 784]
    )
 
 2. Generate Swift code that documents the architecture:
 
    let swiftCode = model.generateCoreMLSwiftCode(inputShape: [1, 784])
    print(swiftCode)
 
 3. Load the specification back:
 
    let spec = try CoreMLExport.loadSpecification(
        from: URL(fileURLWithPath: "model_spec.json")
    )
    // Access weights: spec["layers"] contains layer configurations
 
 4. For production CoreML deployment, you have several options:
 
    Option A: Use CreateML (recommended for new models)
    - Train directly in CreateML using your data
    - Produces optimized .mlmodel files
    
    Option B: Use Core ML Tools in a Swift script
    - Import the JSON specification
    - Use CoreML's MLModelDescription APIs
    
    Option C: Use the weights in your own CoreML model
    - Extract weights from the JSON
    - Apply them to a CoreML model built with MLModelDescription
 
 5. Use the CoreML model in your iOS/macOS app:
 
    import CoreML
    
    // Load the compiled model
    let config = MLModelConfiguration()
    let model = try MLModel(contentsOf: modelURL, configuration: config)
    
    // Create input
    let inputArray = try MLMultiArray(shape: [1, 784], dataType: .float32)
    // Fill inputArray with your data...
    
    // Make prediction
    let input = try MLDictionaryFeatureProvider(
        dictionary: ["input": MLFeatureValue(multiArray: inputArray)]
    )
    let output = try model.prediction(from: input)
*/
