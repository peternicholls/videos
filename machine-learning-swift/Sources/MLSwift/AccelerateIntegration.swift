/// AccelerateIntegration.swift
/// Integration with Apple's Accelerate framework for optimized operations
/// Provides high-performance vector and matrix operations using vDSP and BLAS
/// Only available on Apple platforms (macOS, iOS, tvOS, watchOS)

import Foundation

#if canImport(Accelerate)
import Accelerate

// MARK: - vDSP Vector Operations

/// Normalize data using Accelerate's vDSP (optimized for Apple Silicon)
/// - Parameter matrix: Matrix to normalize
/// - Returns: Tuple of (normalized matrix, mean, standard deviation)
public func normalizeWithAccelerate(_ matrix: Matrix) -> (normalized: Matrix, mean: Float, stdDev: Float) {
    var mean: Float = 0.0
    var stdDev: Float = 0.0
    
    // Use withUnsafeBufferPointer for thread-safe access
    matrix.data.withUnsafeBufferPointer { dataPtr in
        // Compute mean using Accelerate
        vDSP_meanv(dataPtr.baseAddress!, 1, &mean, vDSP_Length(dataPtr.count))
    }
    
    // Compute standard deviation
    var subtracted = [Float](repeating: 0.0, count: matrix.data.count)
    var negMean = -mean
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        subtracted.withUnsafeMutableBufferPointer { subPtr in
            vDSP_vsadd(dataPtr.baseAddress!, 1, &negMean, subPtr.baseAddress!, 1, vDSP_Length(dataPtr.count))
        }
    }
    
    var sumOfSquares: Float = 0.0
    subtracted.withUnsafeBufferPointer { subPtr in
        vDSP_svesq(subPtr.baseAddress!, 1, &sumOfSquares, vDSP_Length(subPtr.count))
    }
    stdDev = sqrt(sumOfSquares / Float(matrix.data.count))
    
    // Normalize: (x - mean) / stdDev
    var normalized = [Float](repeating: 0.0, count: matrix.data.count)
    
    if stdDev > 0 {
        var invStdDev = 1.0 / stdDev
        subtracted.withUnsafeBufferPointer { subPtr in
            normalized.withUnsafeMutableBufferPointer { normPtr in
                vDSP_vsmul(subPtr.baseAddress!, 1, &invStdDev, normPtr.baseAddress!, 1, vDSP_Length(subPtr.count))
            }
        }
    } else {
        normalized = subtracted
    }
    
    return (Matrix(rows: matrix.rows, cols: matrix.cols, data: normalized), mean, stdDev)
}

/// Scale vector by a constant using vDSP
/// - Parameters:
///   - matrix: Matrix to scale
///   - scalar: Scaling factor
/// - Returns: Scaled matrix
public func scaleWithAccelerate(_ matrix: Matrix, by scalar: Float) -> Matrix {
    var result = [Float](repeating: 0.0, count: matrix.data.count)
    var scaleFactor = scalar
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        result.withUnsafeMutableBufferPointer { resultPtr in
            vDSP_vsmul(dataPtr.baseAddress!, 1, &scaleFactor, resultPtr.baseAddress!, 1, vDSP_Length(dataPtr.count))
        }
    }
    
    return Matrix(rows: matrix.rows, cols: matrix.cols, data: result)
}

/// Add two matrices using vDSP
/// - Parameters:
///   - a: First matrix
///   - b: Second matrix
/// - Returns: Sum of matrices
public func addWithAccelerate(_ a: Matrix, _ b: Matrix) -> Matrix? {
    guard a.rows == b.rows && a.cols == b.cols else {
        return nil
    }
    
    var result = [Float](repeating: 0.0, count: a.data.count)
    
    a.data.withUnsafeBufferPointer { aPtr in
        b.data.withUnsafeBufferPointer { bPtr in
            result.withUnsafeMutableBufferPointer { resultPtr in
                vDSP_vadd(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, resultPtr.baseAddress!, 1, vDSP_Length(a.data.count))
            }
        }
    }
    
    return Matrix(rows: a.rows, cols: a.cols, data: result)
}

/// Compute dot product using vDSP
/// - Parameters:
///   - a: First vector
///   - b: Second vector
/// - Returns: Dot product
public func dotProductWithAccelerate(_ a: [Float], _ b: [Float]) -> Float? {
    guard a.count == b.count else {
        return nil
    }
    
    var result: Float = 0.0
    
    a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            vDSP_dotpr(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, &result, vDSP_Length(a.count))
        }
    }
    
    return result
}

// MARK: - BLAS Matrix Operations

/// Matrix multiplication using BLAS (optimized for Apple Silicon)
/// - Parameters:
///   - a: First matrix (m × k)
///   - b: Second matrix (k × n)
/// - Returns: Product matrix (m × n)
public func matrixMultiplyWithBLAS(_ a: Matrix, _ b: Matrix) -> Matrix? {
    // Ensure compatible dimensions
    guard a.cols == b.rows else {
        return nil
    }
    
    var result = [Float](repeating: 0.0, count: a.rows * b.cols)
    
    // Use withUnsafeBufferPointer for safe memory access
    a.data.withUnsafeBufferPointer { aPtr in
        b.data.withUnsafeBufferPointer { bPtr in
            result.withUnsafeMutableBufferPointer { resultPtr in
                // cblas_sgemm performs: C = alpha*A*B + beta*C
                // Using row-major order (CblasRowMajor)
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    Int32(a.rows), Int32(b.cols), Int32(a.cols),
                    1.0,  // alpha
                    aPtr.baseAddress!, Int32(a.cols),
                    bPtr.baseAddress!, Int32(b.cols),
                    0.0,  // beta
                    resultPtr.baseAddress!, Int32(b.cols)
                )
            }
        }
    }
    
    return Matrix(rows: a.rows, cols: b.cols, data: result)
}

/// Matrix-vector multiplication using BLAS
/// - Parameters:
///   - matrix: Matrix (m × n)
///   - vector: Vector (n × 1)
/// - Returns: Result vector (m × 1)
public func matrixVectorMultiplyWithBLAS(_ matrix: Matrix, _ vector: Matrix) -> Matrix? {
    guard matrix.cols == vector.rows && vector.cols == 1 else {
        return nil
    }
    
    var result = [Float](repeating: 0.0, count: matrix.rows)
    
    matrix.data.withUnsafeBufferPointer { matPtr in
        vector.data.withUnsafeBufferPointer { vecPtr in
            result.withUnsafeMutableBufferPointer { resultPtr in
                // cblas_sgemv performs: y = alpha*A*x + beta*y
                cblas_sgemv(
                    CblasRowMajor,
                    CblasNoTrans,
                    Int32(matrix.rows), Int32(matrix.cols),
                    1.0,  // alpha
                    matPtr.baseAddress!, Int32(matrix.cols),
                    vecPtr.baseAddress!, 1,
                    0.0,  // beta
                    resultPtr.baseAddress!, 1
                )
            }
        }
    }
    
    return Matrix(rows: matrix.rows, cols: 1, data: result)
}

// MARK: - Statistics with vDSP

/// Compute mean of matrix using vDSP
/// - Parameter matrix: Matrix
/// - Returns: Mean value
public func meanWithAccelerate(_ matrix: Matrix) -> Float {
    var mean: Float = 0.0
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        vDSP_meanv(dataPtr.baseAddress!, 1, &mean, vDSP_Length(dataPtr.count))
    }
    
    return mean
}

/// Compute sum of matrix using vDSP
/// - Parameter matrix: Matrix
/// - Returns: Sum of all elements
public func sumWithAccelerate(_ matrix: Matrix) -> Float {
    var sum: Float = 0.0
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        vDSP_sve(dataPtr.baseAddress!, 1, &sum, vDSP_Length(dataPtr.count))
    }
    
    return sum
}

/// Find maximum value using vDSP
/// - Parameter matrix: Matrix
/// - Returns: Maximum value
public func maxWithAccelerate(_ matrix: Matrix) -> Float {
    var maxVal: Float = 0.0
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        vDSP_maxv(dataPtr.baseAddress!, 1, &maxVal, vDSP_Length(dataPtr.count))
    }
    
    return maxVal
}

/// Find minimum value using vDSP
/// - Parameter matrix: Matrix
/// - Returns: Minimum value
public func minWithAccelerate(_ matrix: Matrix) -> Float {
    var minVal: Float = 0.0
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        vDSP_minv(dataPtr.baseAddress!, 1, &minVal, vDSP_Length(dataPtr.count))
    }
    
    return minVal
}

#endif // canImport(Accelerate)

