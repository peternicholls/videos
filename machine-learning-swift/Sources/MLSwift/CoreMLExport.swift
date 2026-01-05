/// CoreMLExport.swift
/// Export MLSwift models to CoreML format for iOS/macOS deployment
/// Note: This provides a foundation for CoreML export; full implementation
/// requires CoreML tools which may not be available in all environments

import Foundation

// MARK: - CoreML Export

/// CoreML model export utilities
public enum CoreMLExport {
    
    /// Export a sequential model to a CoreML-compatible JSON specification
    /// This can be used with Python's coremltools to generate the .mlmodel file
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
    
    /// Save the model specification to a file
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
}

// MARK: - CoreML Export Errors

public enum CoreMLExportError: Error, LocalizedError {
    case unsupportedLayerType(String)
    case jsonEncodingFailed
    case invalidModelStructure
    
    public var errorDescription: String? {
        switch self {
        case .unsupportedLayerType(let type):
            return "Unsupported layer type for CoreML export: \(type)"
        case .jsonEncodingFailed:
            return "Failed to encode model to JSON"
        case .invalidModelStructure:
            return "Model structure is invalid for CoreML export"
        }
    }
}

// MARK: - Python Script Generator

extension CoreMLExport {
    
    /// Generate a Python script that converts the JSON spec to a CoreML model
    /// - Parameters:
    ///   - jsonPath: Path to the JSON specification file
    ///   - outputPath: Path for the output .mlmodel file
    /// - Returns: Python script as a string
    public static func generatePythonConverter(
        jsonPath: String,
        outputPath: String
    ) -> String {
        return """
        #!/usr/bin/env python3
        \"\"\"
        MLSwift to CoreML Converter
        Generated automatically - do not edit
        
        Requirements:
            pip install coremltools numpy
        \"\"\"
        
        import json
        import numpy as np
        import coremltools as ct
        from coremltools.models.neural_network import NeuralNetworkBuilder
        from coremltools.models import datatypes
        
        def convert_mlswift_to_coreml(json_path, output_path):
            # Load the JSON specification
            with open(json_path, 'r') as f:
                spec = json.load(f)
            
            # Get input/output info
            input_name = spec['inputName']
            input_shape = spec['inputShape']
            output_name = spec['outputName']
            
            # Create input features
            input_features = [(input_name, datatypes.Array(*input_shape))]
            output_features = [(output_name, None)]
            
            # Create builder
            builder = NeuralNetworkBuilder(
                input_features,
                output_features,
                disable_rank5_shape_mapping=True
            )
            
            # Add layers
            prev_layer = input_name
            for layer in spec['layers']:
                layer_type = layer['type']
                layer_name = layer['name']
                
                if layer_type == 'innerProduct':
                    weights = np.array(layer['weights']).reshape(
                        layer['outputChannels'], layer['inputChannels']
                    )
                    bias = np.array(layer['bias'])
                    builder.add_inner_product(
                        name=layer_name,
                        W=weights.T,  # CoreML expects transposed weights
                        b=bias,
                        input_channels=layer['inputChannels'],
                        output_channels=layer['outputChannels'],
                        has_bias=layer['hasBias'],
                        input_name=prev_layer,
                        output_name=layer_name
                    )
                    prev_layer = layer_name
                    
                elif layer_type == 'activation':
                    act_type = layer['activationType']
                    if act_type == 'ReLU':
                        builder.add_activation(
                            name=layer_name,
                            non_linearity='RELU',
                            input_name=prev_layer,
                            output_name=layer_name
                        )
                    elif act_type == 'sigmoid':
                        builder.add_activation(
                            name=layer_name,
                            non_linearity='SIGMOID',
                            input_name=prev_layer,
                            output_name=layer_name
                        )
                    elif act_type == 'tanh':
                        builder.add_activation(
                            name=layer_name,
                            non_linearity='TANH',
                            input_name=prev_layer,
                            output_name=layer_name
                        )
                    prev_layer = layer_name
                    
                elif layer_type == 'softmax':
                    builder.add_softmax(
                        name=layer_name,
                        input_name=prev_layer,
                        output_name=layer_name
                    )
                    prev_layer = layer_name
                    
                elif layer_type == 'batchnorm':
                    gamma = np.array(layer['gamma'])
                    beta = np.array(layer['beta'])
                    mean = np.array(layer['mean'])
                    variance = np.array(layer['variance'])
                    builder.add_batchnorm(
                        name=layer_name,
                        channels=layer['channels'],
                        gamma=gamma,
                        beta=beta,
                        mean=mean,
                        variance=variance,
                        input_name=prev_layer,
                        output_name=layer_name
                    )
                    prev_layer = layer_name
            
            # Set output
            builder.add_copy(
                name='output_copy',
                input_name=prev_layer,
                output_name=output_name
            )
            
            # Build and save
            mlmodel = ct.models.MLModel(builder.spec)
            mlmodel.save(output_path)
            print(f"CoreML model saved to {output_path}")
        
        if __name__ == '__main__':
            convert_mlswift_to_coreml('\(jsonPath)', '\(outputPath)')
        """
    }
}

// MARK: - Model Extension for Export

extension SequentialModel {
    
    /// Export model to CoreML-compatible format
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
    
    /// Generate a Python conversion script for this model
    /// - Parameters:
    ///   - jsonPath: Path where the JSON spec will be saved
    ///   - outputPath: Desired path for the .mlmodel output
    /// - Returns: Python script string
    public func generateCoreMLConverter(jsonPath: String, outputPath: String) -> String {
        return CoreMLExport.generatePythonConverter(
            jsonPath: jsonPath,
            outputPath: outputPath
        )
    }
}

// MARK: - Usage Example

/*
 Usage:
 
 1. Export your trained MLSwift model:
 
    let model = SequentialModel()
    // ... add layers and train ...
    
    // Export to JSON specification
    try model.exportForCoreML(to: URL(fileURLWithPath: "model_spec.json"), inputShape: [1, 784])
    
    // Generate Python converter script
    let script = model.generateCoreMLConverter(
        jsonPath: "model_spec.json",
        outputPath: "MyModel.mlmodel"
    )
    try script.write(to: URL(fileURLWithPath: "convert.py"), atomically: true, encoding: .utf8)
 
 2. Run the Python script to generate the .mlmodel file:
 
    $ pip install coremltools numpy
    $ python convert.py
 
 3. Add MyModel.mlmodel to your Xcode project and use it:
 
    import CoreML
    
    let model = try MyModel()
    let input = MyModelInput(input: inputArray)
    let output = try model.prediction(input: input)
*/
