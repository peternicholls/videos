/// DataLoader.swift
/// Utilities for loading and saving datasets
/// Compatible with Python's numpy format for seamless data exchange

import Foundation

/// Errors that can occur during data loading
public enum DataLoaderError: Error, CustomStringConvertible {
    case fileSizeMismatch(expected: Int, actual: Int)
    case invalidAlignment
    case fileNotFound(String)
    case invalidDimensions
    
    public var description: String {
        switch self {
        case .fileSizeMismatch(let expected, let actual):
            return "File size mismatch: expected \(expected) bytes, got \(actual) bytes"
        case .invalidAlignment:
            return "Data is not properly aligned for Float access"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidDimensions:
            return "Invalid matrix dimensions"
        }
    }
}

/// Load binary matrix data from file (compatible with Python's .tofile())
/// - Parameters:
///   - url: URL of the file to load
///   - rows: Number of rows in the matrix
///   - cols: Number of columns in the matrix
/// - Returns: Matrix loaded from file
/// - Throws: DataLoaderError if file cannot be loaded or dimensions don't match
public func loadBinaryMatrix(from url: URL, rows: Int, cols: Int) throws -> Matrix {
    let data = try Data(contentsOf: url)
    let floatCount = rows * cols
    
    // Ensure data size matches expected dimensions
    guard data.count == floatCount * MemoryLayout<Float>.size else {
        throw DataLoaderError.fileSizeMismatch(
            expected: floatCount * MemoryLayout<Float>.size,
            actual: data.count
        )
    }
    
    // Ensure data is properly aligned for Float access
    guard data.count % MemoryLayout<Float>.alignment == 0 else {
        throw DataLoaderError.invalidAlignment
    }
    
    // Convert Data to [Float] using safe buffer access
    let floatArray = data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [Float] in
        let floatBuffer = buffer.bindMemory(to: Float.self)
        return Array(floatBuffer)
    }
    
    return Matrix(rows: rows, cols: cols, data: floatArray)
}

/// Save matrix data to binary file (compatible with Python's .tofile())
/// - Parameters:
///   - matrix: Matrix to save
///   - url: URL where the file should be saved
/// - Throws: Error if file cannot be written
public func saveBinaryMatrix(_ matrix: Matrix, to url: URL) throws {
    let data = Data(bytes: matrix.data, count: matrix.data.count * MemoryLayout<Float>.size)
    try data.write(to: url)
}

/// Load multiple matrices from a directory
/// - Parameters:
///   - directory: Directory containing binary matrix files
///   - pattern: File name pattern (e.g., "*.mat")
///   - rows: Number of rows per matrix
///   - cols: Number of columns per matrix
/// - Returns: Array of loaded matrices
public func loadMatricesFromDirectory(
    _ directory: URL,
    pattern: String = "*.mat",
    rows: Int,
    cols: Int
) throws -> [Matrix] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    
    var matrices: [Matrix] = []
    for fileURL in files where fileURL.pathExtension == "mat" {
        let matrix = try loadBinaryMatrix(from: fileURL, rows: rows, cols: cols)
        matrices.append(matrix)
    }
    
    return matrices
}
