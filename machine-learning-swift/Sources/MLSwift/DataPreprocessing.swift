/// DataPreprocessing.swift
/// Data preprocessing utilities for machine learning
/// Includes normalization, standardization, encoding, and dataset splitting

import Foundation

// MARK: - Normalization

/// Normalize matrix values to [0, 1] range
/// - Parameters:
///   - matrix: Matrix to normalize
///   - maxValue: Maximum value in the original scale (default: 255.0 for images)
/// - Returns: Normalized matrix
public func normalizeToRange(_ matrix: Matrix, maxValue: Float = 255.0) -> Matrix {
    var result = matrix
    for i in 0..<result.data.count {
        result.data[i] /= maxValue
    }
    return result
}

/// Min-Max normalization to [0, 1] range
/// - Parameter matrix: Matrix to normalize
/// - Returns: Normalized matrix with values in [0, 1]
public func minMaxNormalize(_ matrix: Matrix) -> Matrix {
    guard let minVal = matrix.data.min(), let maxVal = matrix.data.max() else {
        return matrix
    }
    
    let range = maxVal - minVal
    guard range > 0 else { return matrix }
    
    var result = matrix
    for i in 0..<result.data.count {
        result.data[i] = (result.data[i] - minVal) / range
    }
    return result
}

/// Standardize data (zero mean, unit variance)
/// - Parameter matrix: Matrix to standardize
/// - Returns: Tuple of (standardized matrix, mean, standard deviation)
public func standardize(_ matrix: Matrix) -> (normalized: Matrix, mean: Float, stdDev: Float) {
    let count = Float(matrix.data.count)
    let mean = matrix.data.reduce(0, +) / count
    
    let variance = matrix.data.map { pow($0 - mean, 2) }.reduce(0, +) / count
    let stdDev = sqrt(variance)
    
    guard stdDev > 0 else {
        return (matrix, mean, 0)
    }
    
    var result = matrix
    for i in 0..<result.data.count {
        result.data[i] = (result.data[i] - mean) / stdDev
    }
    
    return (result, mean, stdDev)
}

// MARK: - Label Encoding

/// Convert integer labels to one-hot encoded vectors
/// - Parameters:
///   - labels: Array of integer class labels
///   - numClasses: Total number of classes
/// - Returns: Array of one-hot encoded matrices
public func oneHotEncode(labels: [Int], numClasses: Int) -> [Matrix] {
    return labels.map { label in
        var encoded = [Float](repeating: 0.0, count: numClasses)
        if label >= 0 && label < numClasses {
            encoded[label] = 1.0
        }
        return Matrix(rows: numClasses, cols: 1, data: encoded)
    }
}

/// Convert one-hot encoded vectors back to integer labels
/// - Parameter encoded: Array of one-hot encoded matrices
/// - Returns: Array of integer labels
public func oneHotDecode(_ encoded: [Matrix]) -> [Int] {
    return encoded.map { matrix in
        var maxIndex = 0
        var maxValue = matrix.data[0]
        
        for i in 1..<matrix.data.count {
            if matrix.data[i] > maxValue {
                maxValue = matrix.data[i]
                maxIndex = i
            }
        }
        
        return maxIndex
    }
}

// MARK: - Dataset Utilities

/// Split dataset into training and validation sets
/// - Parameters:
///   - data: Array of data samples
///   - splitRatio: Ratio of training data (default: 0.8 for 80/20 split)
/// - Returns: Tuple of (training data, validation data)
public func trainValidationSplit<T>(data: [T], splitRatio: Float = 0.8) -> (train: [T], validation: [T]) {
    guard splitRatio > 0 && splitRatio < 1 else {
        return (data, [])
    }
    
    let trainCount = Int(Float(data.count) * splitRatio)
    let trainData = Array(data[..<trainCount])
    let validationData = Array(data[trainCount...])
    
    return (trainData, validationData)
}

/// Shuffle two arrays while maintaining correspondence
/// - Parameters:
///   - inputs: First array (e.g., features)
///   - targets: Second array (e.g., labels)
/// - Returns: Tuple of shuffled arrays
public func shuffleDataset(inputs: [Matrix], targets: [Matrix]) -> (inputs: [Matrix], targets: [Matrix]) {
    guard inputs.count == targets.count else {
        return (inputs, targets)
    }
    
    var indices = Array(0..<inputs.count)
    indices.shuffle()
    
    let shuffledInputs = indices.map { inputs[$0] }
    let shuffledTargets = indices.map { targets[$0] }
    
    return (shuffledInputs, shuffledTargets)
}

/// Create mini-batches from dataset
/// - Parameters:
///   - inputs: Array of input matrices
///   - targets: Array of target matrices
///   - batchSize: Size of each batch
/// - Returns: Array of (inputs, targets) tuples for each batch
public func createBatches(
    inputs: [Matrix],
    targets: [Matrix],
    batchSize: Int
) -> [(inputs: [Matrix], targets: [Matrix])] {
    guard inputs.count == targets.count else {
        return []
    }
    
    var batches: [(inputs: [Matrix], targets: [Matrix])] = []
    
    for i in stride(from: 0, to: inputs.count, by: batchSize) {
        let endIdx = min(i + batchSize, inputs.count)
        let batchInputs = Array(inputs[i..<endIdx])
        let batchTargets = Array(targets[i..<endIdx])
        batches.append((batchInputs, batchTargets))
    }
    
    return batches
}

// MARK: - Image Processing

/// Flatten image matrices for neural network input
/// - Parameter images: Array of 2D image matrices
/// - Returns: Array of flattened 1D matrices
public func flattenImages(_ images: [Matrix]) -> [Matrix] {
    return images.map { image in
        Matrix(rows: image.rows * image.cols, cols: 1, data: image.data)
    }
}

/// Reshape flattened vectors back to images
/// - Parameters:
///   - flattened: Array of flattened matrices
///   - height: Image height
///   - width: Image width
/// - Returns: Array of 2D image matrices
public func reshapeToImages(_ flattened: [Matrix], height: Int, width: Int) -> [Matrix] {
    return flattened.map { flat in
        Matrix(rows: height, cols: width, data: flat.data)
    }
}
