/// MetalDevice.swift
/// Metal device manager for GPU-accelerated operations
/// Optimized for macOS with Apple Silicon

import Foundation
#if canImport(Metal)
import Metal

/// Manages Metal device and compute pipeline for neural network operations
public class MetalDevice {
    /// Shared singleton instance
    public static let shared = MetalDevice()
    
    /// Metal device (Apple Silicon GPU)
    public let device: MTLDevice
    
    /// Command queue for executing GPU operations
    public let commandQueue: MTLCommandQueue
    
    /// Compute pipeline states for various operations
    private var pipelineStates: [String: MTLComputePipelineState] = [:]
    
    /// Default library containing Metal shaders
    private let library: MTLLibrary
    
    /// Private initializer for singleton
    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = commandQueue
        
        // Load Metal library from source
        do {
            guard let library = try device.makeDefaultLibrary() else {
                fatalError("Failed to create Metal library: Library is nil")
            }
            self.library = library
        } catch {
            fatalError("Failed to create Metal library: \(error.localizedDescription)")
        }
        
        // Pre-compile commonly used compute pipelines
        compileCommonPipelines()
    }
    
    /// Compile commonly used compute pipeline states
    private func compileCommonPipelines() {
        let kernelNames = [
            "matrix_add",
            "matrix_subtract",
            "matrix_scale",
            "matrix_multiply",
            "matrix_multiply_transpose_a",
            "matrix_multiply_transpose_b",
            "relu_forward",
            "relu_backward",
            "softmax_max_reduce",
            "softmax_exp_sum",
            "softmax_normalize",
            "cross_entropy_forward",
            "cross_entropy_backward_q",
            "cross_entropy_backward_p"
        ]
        
        for name in kernelNames {
            do {
                let pipelineState = try createPipelineState(for: name)
                pipelineStates[name] = pipelineState
            } catch {
                print("Warning: Failed to compile pipeline '\(name)': \(error)")
            }
        }
    }
    
    /// Create a compute pipeline state for a given kernel function
    /// - Parameter kernelName: Name of the Metal kernel function
    /// - Returns: Compiled compute pipeline state
    private func createPipelineState(for kernelName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: kernelName) else {
            throw MetalError.functionNotFound(kernelName)
        }
        return try device.makeComputePipelineState(function: function)
    }
    
    /// Get or create a compute pipeline state
    /// - Parameter name: Kernel function name
    /// - Returns: Compute pipeline state
    public func getPipelineState(_ name: String) throws -> MTLComputePipelineState {
        if let cached = pipelineStates[name] {
            return cached
        }
        let state = try createPipelineState(for: name)
        pipelineStates[name] = state
        return state
    }
    
    /// Create a Metal buffer from Float array
    /// - Parameter data: Float array data
    /// - Returns: Metal buffer
    public func makeBuffer(from data: [Float]) -> MTLBuffer? {
        let size = data.count * MemoryLayout<Float>.stride
        return device.makeBuffer(bytes: data, length: size, options: .storageModeShared)
    }
    
    /// Create an empty Metal buffer
    /// - Parameter count: Number of Float elements
    /// - Returns: Metal buffer
    public func makeBuffer(count: Int) -> MTLBuffer? {
        let size = count * MemoryLayout<Float>.stride
        return device.makeBuffer(length: size, options: .storageModeShared)
    }
    
    /// Read Float data from Metal buffer
    /// - Parameters:
    ///   - buffer: Metal buffer to read from
    ///   - count: Number of Float elements to read
    /// - Returns: Array of Float values
    public func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    /// Execute a compute kernel with 1D thread grid
    /// - Parameters:
    ///   - kernelName: Name of kernel function
    ///   - buffers: Array of Metal buffers to bind
    ///   - threadCount: Number of threads to dispatch
    public func execute1D(
        kernel kernelName: String,
        buffers: [MTLBuffer?],
        threadCount: Int
    ) throws {
        let pipelineState = try getPipelineState(kernelName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandCreationFailed
        }
        
        encoder.setComputePipelineState(pipelineState)
        
        for (index, buffer) in buffers.enumerated() {
            if let buffer = buffer {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
        }
        
        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, threadCount)
        let threadGroups = (threadCount + threadGroupSize - 1) / threadGroupSize
        
        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Execute a compute kernel with 2D thread grid
    /// - Parameters:
    ///   - kernelName: Name of kernel function
    ///   - buffers: Array of Metal buffers to bind
    ///   - gridSize: 2D grid size (width, height)
    public func execute2D(
        kernel kernelName: String,
        buffers: [MTLBuffer?],
        gridSize: (width: Int, height: Int)
    ) throws {
        let pipelineState = try getPipelineState(kernelName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandCreationFailed
        }
        
        encoder.setComputePipelineState(pipelineState)
        
        for (index, buffer) in buffers.enumerated() {
            if let buffer = buffer {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
        }
        
        // Use 16x16 thread groups (common for 2D operations)
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (gridSize.width + 15) / 16,
            height: (gridSize.height + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

/// Metal-specific errors
public enum MetalError: Error {
    case functionNotFound(String)
    case commandCreationFailed
    case bufferCreationFailed
    case invalidDimensions
}

#endif  // canImport(Metal)
