/// Matrix.swift
/// Core matrix type for neural network operations
///
/// This file defines the Matrix type which represents 2D matrices used throughout
/// the neural network library. Matrices are stored in row-major order and support
/// both CPU and Metal GPU operations.

import Foundation
#if canImport(Metal)
import Metal
#endif

/// A 2D matrix with float32 elements stored in row-major order
///
/// The Matrix type provides efficient storage and operations for neural network
/// computations. It supports both CPU-based operations and GPU acceleration via Metal.
///
/// - Note: Element (row, col) is stored at index `row * cols + col`
public struct Matrix {
    /// Number of rows in the matrix
    public let rows: Int
    
    /// Number of columns in the matrix
    public let cols: Int
    
    /// Matrix data in row-major order
    public var data: [Float]
    
    /// Total number of elements (rows * cols)
    public var count: Int {
        rows * cols
    }
    
    // MARK: - Initialization
    
    /// Create a zero-initialized matrix
    /// - Parameters:
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    public init(rows: Int, cols: Int) {
        precondition(rows > 0 && cols > 0, "Matrix dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: 0.0, count: rows * cols)
    }
    
    /// Create a matrix with specified data
    /// - Parameters:
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - data: Initial data (must have rows * cols elements)
    public init(rows: Int, cols: Int, data: [Float]) {
        precondition(rows > 0 && cols > 0, "Matrix dimensions must be positive")
        precondition(data.count == rows * cols, "Data size must match rows * cols")
        self.rows = rows
        self.cols = cols
        self.data = data
    }
    
    /// Create a matrix filled with a constant value
    /// - Parameters:
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - value: Value to fill with
    public init(rows: Int, cols: Int, value: Float) {
        precondition(rows > 0 && cols > 0, "Matrix dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: value, count: rows * cols)
    }
    
    /// Create a matrix with random values in a specified range
    /// - Parameters:
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - lower: Lower bound (inclusive)
    ///   - upper: Upper bound (exclusive)
    public init(rows: Int, cols: Int, randomInRange lower: Float, _ upper: Float) {
        precondition(rows > 0 && cols > 0, "Matrix dimensions must be positive")
        precondition(lower < upper, "Lower bound must be less than upper bound")
        self.rows = rows
        self.cols = cols
        let range = upper - lower
        self.data = (0..<(rows * cols)).map { _ in
            Float.random(in: 0..<1) * range + lower
        }
    }
    
    // MARK: - Element Access
    
    /// Access matrix element at (row, col)
    /// - Parameters:
    ///   - row: Row index (0-based)
    ///   - col: Column index (0-based)
    /// - Returns: Element value
    public subscript(row: Int, col: Int) -> Float {
        get {
            precondition(row >= 0 && row < rows, "Row index out of bounds")
            precondition(col >= 0 && col < cols, "Column index out of bounds")
            return data[row * cols + col]
        }
        set {
            precondition(row >= 0 && row < rows, "Row index out of bounds")
            precondition(col >= 0 && col < cols, "Column index out of bounds")
            data[row * cols + col] = newValue
        }
    }
    
    // MARK: - Basic Operations
    
    /// Fill all elements with a constant value
    /// - Parameter value: Value to fill with
    public mutating func fill(_ value: Float) {
        for i in 0..<data.count {
            data[i] = value
        }
    }
    
    /// Zero all elements
    public mutating func zero() {
        fill(0.0)
    }
    
    /// Scale all elements by a constant factor
    /// - Parameter scale: Scaling factor
    public mutating func scale(by scale: Float) {
        for i in 0..<data.count {
            data[i] *= scale
        }
    }
    
    /// Compute the sum of all elements
    /// - Returns: Sum of all matrix elements
    public func sum() -> Float {
        return data.reduce(0.0, +)
    }
    
    /// Find the index of the maximum element
    /// - Returns: Linear index of maximum element
    public func argmax() -> Int {
        guard !data.isEmpty else { return 0 }
        var maxIndex = 0
        var maxValue = data[0]
        for i in 1..<data.count {
            if data[i] > maxValue {
                maxValue = data[i]
                maxIndex = i
            }
        }
        return maxIndex
    }
    
    // MARK: - Matrix Operations
    
    /// Element-wise addition
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    /// - Returns: Result of a + b
    public static func add(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols,
                    "Matrix dimensions must match for addition")
        var result = Matrix(rows: a.rows, cols: a.cols)
        for i in 0..<a.data.count {
            result.data[i] = a.data[i] + b.data[i]
        }
        return result
    }
    
    /// Element-wise subtraction
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    /// - Returns: Result of a - b
    public static func subtract(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols,
                    "Matrix dimensions must match for subtraction")
        var result = Matrix(rows: a.rows, cols: a.cols)
        for i in 0..<a.data.count {
            result.data[i] = a.data[i] - b.data[i]
        }
        return result
    }
    
    /// Matrix multiplication
    /// - Parameters:
    ///   - a: First matrix (M x K)
    ///   - b: Second matrix (K x N)
    ///   - transposeA: If true, use transpose of a
    ///   - transposeB: If true, use transpose of b
    /// - Returns: Result of a * b (M x N)
    public static func multiply(
        _ a: Matrix,
        _ b: Matrix,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) -> Matrix {
        let aRows = transposeA ? a.cols : a.rows
        let aCols = transposeA ? a.rows : a.cols
        let bRows = transposeB ? b.cols : b.rows
        let bCols = transposeB ? b.rows : b.cols
        
        precondition(aCols == bRows,
                    "Inner dimensions must match for multiplication")
        
        var result = Matrix(rows: aRows, cols: bCols)
        
        // Choose the appropriate multiplication variant
        if !transposeA && !transposeB {
            multiplyNN(&result, a, b)
        } else if !transposeA && transposeB {
            multiplyNT(&result, a, b)
        } else if transposeA && !transposeB {
            multiplyTN(&result, a, b)
        } else {
            multiplyTT(&result, a, b)
        }
        
        return result
    }
    
    // MARK: - Matrix Multiplication Variants (optimized for cache locality)
    
    /// Matrix multiply: C = A * B (no transpose)
    private static func multiplyNN(_ c: inout Matrix, _ a: Matrix, _ b: Matrix) {
        for i in 0..<c.rows {
            for k in 0..<a.cols {
                let aVal = a.data[i * a.cols + k]
                for j in 0..<c.cols {
                    c.data[i * c.cols + j] += aVal * b.data[k * b.cols + j]
                }
            }
        }
    }
    
    /// Matrix multiply: C = A * B^T (transpose B)
    private static func multiplyNT(_ c: inout Matrix, _ a: Matrix, _ b: Matrix) {
        for i in 0..<c.rows {
            for j in 0..<c.cols {
                var sum: Float = 0.0
                for k in 0..<a.cols {
                    sum += a.data[i * a.cols + k] * b.data[j * b.cols + k]
                }
                c.data[i * c.cols + j] = sum
            }
        }
    }
    
    /// Matrix multiply: C = A^T * B (transpose A)
    private static func multiplyTN(_ c: inout Matrix, _ a: Matrix, _ b: Matrix) {
        for k in 0..<a.rows {
            for i in 0..<c.rows {
                let aVal = a.data[k * a.cols + i]
                for j in 0..<c.cols {
                    c.data[i * c.cols + j] += aVal * b.data[k * b.cols + j]
                }
            }
        }
    }
    
    /// Matrix multiply: C = A^T * B^T (transpose both)
    private static func multiplyTT(_ c: inout Matrix, _ a: Matrix, _ b: Matrix) {
        for i in 0..<c.rows {
            for j in 0..<c.cols {
                var sum: Float = 0.0
                for k in 0..<a.rows {
                    sum += a.data[k * a.cols + i] * b.data[j * b.cols + k]
                }
                c.data[i * c.cols + j] = sum
            }
        }
    }
}

// MARK: - CustomStringConvertible

extension Matrix: CustomStringConvertible {
    public var description: String {
        var result = "Matrix(\(rows)x\(cols)):\n"
        for i in 0..<min(rows, 5) {  // Show first 5 rows max
            result += "  ["
            for j in 0..<min(cols, 10) {  // Show first 10 cols max
                result += String(format: "%.4f", self[i, j])
                if j < cols - 1 {
                    result += ", "
                }
            }
            if cols > 10 {
                result += ", ..."
            }
            result += "]\n"
        }
        if rows > 5 {
            result += "  ...\n"
        }
        return result
    }
}
