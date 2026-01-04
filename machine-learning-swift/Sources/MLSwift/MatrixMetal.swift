/// MatrixMetal.swift
/// Metal-accelerated matrix operations extension
/// Optimized for Apple Silicon

import Foundation
#if canImport(Metal)
import Metal

extension Matrix {
    /// Matrix addition using Metal GPU acceleration
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    /// - Returns: Result of a + b
    public static func addGPU(_ a: Matrix, _ b: Matrix) throws -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols,
                    "Matrix dimensions must match for addition")
        
        let device = MetalDevice.shared
        
        guard let bufferA = device.makeBuffer(from: a.data),
              let bufferB = device.makeBuffer(from: b.data),
              let bufferC = device.makeBuffer(count: a.data.count) else {
            throw MetalError.bufferCreationFailed
        }
        
        try device.execute1D(
            kernel: "matrix_add",
            buffers: [bufferA, bufferB, bufferC],
            threadCount: a.data.count
        )
        
        let resultData = device.readBuffer(bufferC, count: a.data.count)
        return Matrix(rows: a.rows, cols: a.cols, data: resultData)
    }
    
    /// Matrix subtraction using Metal GPU acceleration
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    /// - Returns: Result of a - b
    public static func subtractGPU(_ a: Matrix, _ b: Matrix) throws -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols,
                    "Matrix dimensions must match for subtraction")
        
        let device = MetalDevice.shared
        
        guard let bufferA = device.makeBuffer(from: a.data),
              let bufferB = device.makeBuffer(from: b.data),
              let bufferC = device.makeBuffer(count: a.data.count) else {
            throw MetalError.bufferCreationFailed
        }
        
        try device.execute1D(
            kernel: "matrix_subtract",
            buffers: [bufferA, bufferB, bufferC],
            threadCount: a.data.count
        )
        
        let resultData = device.readBuffer(bufferC, count: a.data.count)
        return Matrix(rows: a.rows, cols: a.cols, data: resultData)
    }
    
    /// Matrix multiplication using Metal GPU acceleration
    /// - Parameters:
    ///   - a: First matrix (M x K)
    ///   - b: Second matrix (K x N)
    ///   - transposeA: If true, use transpose of a
    ///   - transposeB: If true, use transpose of b
    /// - Returns: Result of a * b (M x N)
    public static func multiplyGPU(
        _ a: Matrix,
        _ b: Matrix,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) throws -> Matrix {
        let device = MetalDevice.shared
        
        let aRows = transposeA ? a.cols : a.rows
        let aCols = transposeA ? a.rows : a.cols
        let bRows = transposeB ? b.cols : b.rows
        let bCols = transposeB ? b.rows : b.cols
        
        precondition(aCols == bRows,
                    "Inner dimensions must match for multiplication")
        
        let M = UInt32(aRows)
        let K = UInt32(aCols)
        let N = UInt32(bCols)
        
        guard let bufferA = device.makeBuffer(from: a.data),
              let bufferB = device.makeBuffer(from: b.data),
              let bufferC = device.makeBuffer(count: aRows * bCols) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create buffers for dimension parameters
        var mParam = M
        var kParam = K
        var nParam = N
        
        guard let bufferM = device.device.makeBuffer(
                bytes: &mParam,
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let bufferK = device.device.makeBuffer(
                bytes: &kParam,
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let bufferN = device.device.makeBuffer(
                bytes: &nParam,
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Choose kernel based on transpose flags
        let kernelName: String
        if !transposeA && !transposeB {
            kernelName = "matrix_multiply"
        } else if transposeA && !transposeB {
            kernelName = "matrix_multiply_transpose_a"
        } else if !transposeA && transposeB {
            kernelName = "matrix_multiply_transpose_b"
        } else {
            // Both transposeA and transposeB are true
            // Using the basic matrix_multiply kernel without actually transposing
            // the inputs would produce incorrect results. Fail fast instead.
            preconditionFailure("Matrix.multiplyGPU does not support transposeA && transposeB; transpose inputs explicitly or add a dedicated double-transpose kernel.")
        }
        
        try device.execute2D(
            kernel: kernelName,
            buffers: [bufferA, bufferB, bufferC, bufferM, bufferK, bufferN],
            gridSize: (width: Int(N), height: Int(M))
        )
        
        let resultData = device.readBuffer(bufferC, count: aRows * bCols)
        return Matrix(rows: aRows, cols: bCols, data: resultData)
    }
    
    /// Apply ReLU activation using Metal GPU acceleration
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix with ReLU applied
    public static func reluGPU(_ input: Matrix) throws -> Matrix {
        let device = MetalDevice.shared
        
        guard let bufferIn = device.makeBuffer(from: input.data),
              let bufferOut = device.makeBuffer(count: input.data.count) else {
            throw MetalError.bufferCreationFailed
        }
        
        try device.execute1D(
            kernel: "relu_forward",
            buffers: [bufferIn, bufferOut],
            threadCount: input.data.count
        )
        
        let resultData = device.readBuffer(bufferOut, count: input.data.count)
        return Matrix(rows: input.rows, cols: input.cols, data: resultData)
    }
    
    /// Apply softmax activation using Metal GPU acceleration
    /// - Parameter input: Input matrix (must be a vector: rows=1 or cols=1)
    /// - Returns: Output matrix with softmax applied
    public static func softmaxGPU(_ input: Matrix) throws -> Matrix {
        precondition(input.rows == 1 || input.cols == 1,
                    "Softmax currently only supports vectors")
        
        let device = MetalDevice.shared
        let size = input.data.count
        
        guard let bufferIn = device.makeBuffer(from: input.data),
              let bufferOut = device.makeBuffer(count: size),
              let bufferMax = device.makeBuffer(count: 1),
              let bufferSum = device.makeBuffer(count: 1) else {
            throw MetalError.bufferCreationFailed
        }
        
        var sizeParam = UInt32(size)
        guard let bufferSize = device.device.makeBuffer(
                bytes: &sizeParam,
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Stage 1: Find max
        try device.execute1D(
            kernel: "softmax_max_reduce",
            buffers: [bufferIn, bufferMax, bufferSize],
            threadCount: 256
        )
        
        // Stage 2: Compute exp and sum
        try device.execute1D(
            kernel: "softmax_exp_sum",
            buffers: [bufferIn, bufferOut, bufferSum, bufferMax, bufferSize],
            threadCount: 256
        )
        
        // Stage 3: Normalize
        try device.execute1D(
            kernel: "softmax_normalize",
            buffers: [bufferOut, bufferSum],
            threadCount: size
        )
        
        let resultData = device.readBuffer(bufferOut, count: size)
        return Matrix(rows: input.rows, cols: input.cols, data: resultData)
    }
    
    /// In-place scale using Metal GPU acceleration
    /// - Parameter scale: Scaling factor
    public mutating func scaleGPU(by scale: Float) throws {
        let device = MetalDevice.shared
        
        guard let bufferIn = device.makeBuffer(from: self.data),
              let bufferOut = device.makeBuffer(count: self.data.count) else {
            throw MetalError.bufferCreationFailed
        }
        
        var scaleParam = scale
        guard let bufferScale = device.device.makeBuffer(
                bytes: &scaleParam,
                length: MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        
        try device.execute1D(
            kernel: "matrix_scale",
            buffers: [bufferIn, bufferOut, bufferScale],
            threadCount: self.data.count
        )
        
        self.data = device.readBuffer(bufferOut, count: self.data.count)
    }
}

#endif  // canImport(Metal)
