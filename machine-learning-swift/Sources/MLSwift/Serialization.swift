/// Serialization.swift
/// Model serialization and deserialization for MLSwift
/// Supports saving and loading trained models

import Foundation

/// Model serialization format
public struct ModelData: Codable {
    /// Version of the serialization format
    let version: String
    
    /// Model architecture description
    let architecture: [LayerData]
    
    /// Model metadata
    let metadata: [String: String]
    
    /// Date when model was saved
    let savedDate: Date
}

/// Layer serialization data
public struct LayerData: Codable {
    /// Type of layer (e.g., "Dense", "ReLU", "Softmax")
    let type: String
    
    /// Layer configuration parameters
    let config: [String: String]
    
    /// Layer parameters (weights, biases, etc.)
    let parameters: [MatrixData]
}

/// Matrix serialization data
public struct MatrixData: Codable {
    /// Number of rows
    let rows: Int
    
    /// Number of columns
    let cols: Int
    
    /// Matrix data as flat array
    let data: [Float]
}

/// Extension to add serialization capabilities to Matrix
extension Matrix {
    /// Convert matrix to serializable data
    func toData() -> MatrixData {
        return MatrixData(rows: self.rows, cols: self.cols, data: self.data)
    }
    
    /// Create matrix from serialized data
    static func fromData(_ data: MatrixData) -> Matrix {
        return Matrix(rows: data.rows, cols: data.cols, data: data.data)
    }
}

/// Extension to add serialization to SequentialModel
extension SequentialModel {
    /// Save model to file
    /// - Parameter url: File URL to save the model
    /// - Throws: Serialization or file I/O errors
    public func save(to url: URL) throws {
        var layerDataArray: [LayerData] = []
        
        // Serialize each layer
        for layer in getLayers() {
            let layerData = try serializeLayer(layer)
            layerDataArray.append(layerData)
        }
        
        // Create model data
        let modelData = ModelData(
            version: "1.0",
            architecture: layerDataArray,
            metadata: [
                "framework": "MLSwift",
                "platform": "macOS"
            ],
            savedDate: Date()
        )
        
        // Encode to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let jsonData = try encoder.encode(modelData)
        
        // Write to file
        try jsonData.write(to: url)
    }
    
    /// Load model from file
    /// - Parameter url: File URL to load the model from
    /// - Returns: Loaded SequentialModel
    /// - Throws: Deserialization or file I/O errors
    public static func load(from url: URL) throws -> SequentialModel {
        // Read file
        let jsonData = try Data(contentsOf: url)
        
        // Decode from JSON
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let modelData = try decoder.decode(ModelData.self, from: jsonData)
        
        // Check version compatibility
        guard modelData.version == "1.0" else {
            throw SerializationError.incompatibleVersion(modelData.version)
        }
        
        // Create model
        let model = SequentialModel()
        
        // Deserialize layers
        for layerData in modelData.architecture {
            let layer = try deserializeLayer(layerData)
            model.add(layer)
        }
        
        return model
    }
    
    /// Serialize a layer to LayerData
    private func serializeLayer(_ layer: Layer) throws -> LayerData {
        switch layer {
        case let dense as DenseLayer:
            return LayerData(
                type: "Dense",
                config: [
                    "inputSize": String(dense.weights.cols),
                    "outputSize": String(dense.weights.rows)
                ],
                parameters: [
                    dense.weights.toData(),
                    dense.bias.toData()
                ]
            )
            
        case is ReLULayer:
            return LayerData(
                type: "ReLU",
                config: [:],
                parameters: []
            )
            
        case is SoftmaxLayer:
            return LayerData(
                type: "Softmax",
                config: [:],
                parameters: []
            )
            
        case is SigmoidLayer:
            return LayerData(
                type: "Sigmoid",
                config: [:],
                parameters: []
            )
            
        case is TanhLayer:
            return LayerData(
                type: "Tanh",
                config: [:],
                parameters: []
            )
            
        case let dropout as DropoutLayer:
            return LayerData(
                type: "Dropout",
                config: [
                    "dropoutRate": String(dropout.training ? 0.5 : 0.0) // Note: can't access private dropoutRate
                ],
                parameters: []
            )
            
        case let batchNorm as BatchNormLayer:
            return LayerData(
                type: "BatchNorm",
                config: [:],
                parameters: [
                    batchNorm.gamma.toData(),
                    batchNorm.beta.toData()
                ]
            )
            
        default:
            throw SerializationError.unsupportedLayerType(String(describing: type(of: layer)))
        }
    }
    
    /// Deserialize LayerData to a Layer
    private static func deserializeLayer(_ layerData: LayerData) throws -> Layer {
        switch layerData.type {
        case "Dense":
            guard let inputSizeStr = layerData.config["inputSize"],
                  let outputSizeStr = layerData.config["outputSize"],
                  let inputSize = Int(inputSizeStr),
                  let outputSize = Int(outputSizeStr),
                  layerData.parameters.count == 2 else {
                throw SerializationError.invalidLayerData("Dense layer requires inputSize, outputSize, and 2 parameters")
            }
            
            let layer = DenseLayer(inputSize: inputSize, outputSize: outputSize)
            layer.weights = Matrix.fromData(layerData.parameters[0])
            layer.bias = Matrix.fromData(layerData.parameters[1])
            return layer
            
        case "ReLU":
            return ReLULayer()
            
        case "Softmax":
            return SoftmaxLayer()
            
        case "Sigmoid":
            return SigmoidLayer()
            
        case "Tanh":
            return TanhLayer()
            
        case "Dropout":
            let dropoutRate = Float(layerData.config["dropoutRate"] ?? "0.5") ?? 0.5
            return DropoutLayer(dropoutRate: dropoutRate)
            
        case "BatchNorm":
            guard layerData.parameters.count == 2 else {
                throw SerializationError.invalidLayerData("BatchNorm layer requires 2 parameters")
            }
            
            let gamma = Matrix.fromData(layerData.parameters[0])
            let layer = BatchNormLayer(numFeatures: gamma.rows)
            layer.gamma = gamma
            layer.beta = Matrix.fromData(layerData.parameters[1])
            return layer
            
        default:
            throw SerializationError.unsupportedLayerType(layerData.type)
        }
    }
}

/// Serialization errors
public enum SerializationError: Error, CustomStringConvertible {
    case incompatibleVersion(String)
    case unsupportedLayerType(String)
    case invalidLayerData(String)
    case fileNotFound(String)
    
    public var description: String {
        switch self {
        case .incompatibleVersion(let version):
            return "Incompatible model version: \(version)"
        case .unsupportedLayerType(let type):
            return "Unsupported layer type: \(type)"
        case .invalidLayerData(let message):
            return "Invalid layer data: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        }
    }
}
