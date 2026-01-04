/// Layer.swift
/// Neural network layer abstractions
/// Optimized for Apple Silicon with Metal acceleration

import Foundation

/// Protocol for neural network layers
public protocol Layer {
    /// Forward pass through the layer
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix
    func forward(_ input: Matrix) -> Matrix
    
    /// Backward pass through the layer
    /// - Parameter gradOutput: Gradient from next layer
    /// - Returns: Gradient w.r.t. input
    func backward(_ gradOutput: Matrix) -> Matrix
    
    /// Get trainable parameters
    /// - Returns: Array of parameter matrices
    func parameters() -> [Matrix]
    
    /// Get parameter gradients
    /// - Returns: Array of gradient matrices
    func gradients() -> [Matrix]
    
    /// Update parameters using gradients
    /// - Parameter learningRate: Learning rate for gradient descent
    func updateParameters(learningRate: Float)
    
    /// Scale accumulated gradients by a factor
    /// - Parameter scale: Scaling factor
    func scaleGradients(by scale: Float)
}

/// Fully connected (dense) layer
public class DenseLayer: Layer {
    /// Weight matrix (outputSize x inputSize)
    public var weights: Matrix
    
    /// Bias vector (outputSize x 1)
    public var bias: Matrix
    
    /// Gradient w.r.t. weights
    private var weightGrad: Matrix
    
    /// Gradient w.r.t. bias
    private var biasGrad: Matrix
    
    /// Cached input from forward pass (for backward)
    private var cachedInput: Matrix?
    
    /// Initialize dense layer with Xavier/He initialization
    /// - Parameters:
    ///   - inputSize: Number of input features
    ///   - outputSize: Number of output features
    public init(inputSize: Int, outputSize: Int) {
        // Xavier initialization: uniform in [-bound, bound]
        let bound = sqrt(6.0 / Float(inputSize + outputSize))
        
        self.weights = Matrix(
            rows: outputSize,
            cols: inputSize,
            randomInRange: -bound,
            bound
        )
        self.bias = Matrix(rows: outputSize, cols: 1, value: 0.0)
        
        self.weightGrad = Matrix(rows: outputSize, cols: inputSize)
        self.biasGrad = Matrix(rows: outputSize, cols: 1)
    }
    
    /// Forward pass: output = weights * input + bias
    /// - Parameter input: Input matrix (inputSize x 1)
    /// - Returns: Output matrix (outputSize x 1)
    public func forward(_ input: Matrix) -> Matrix {
        precondition(input.cols == 1, "Input must be a column vector")
        precondition(input.rows == weights.cols, "Input size must match weight columns")
        
        // Cache input for backward pass
        cachedInput = input
        
        // Compute: output = W * x + b
        #if canImport(Metal)
        do {
            let wx = try Matrix.multiplyGPU(weights, input)
            return Matrix.add(wx, bias)
        } catch {
            // Fallback to CPU
            let wx = Matrix.multiply(weights, input)
            return Matrix.add(wx, bias)
        }
        #else
        let wx = Matrix.multiply(weights, input)
        return Matrix.add(wx, bias)
        #endif
    }
    
    /// Backward pass: compute gradients
    /// - Parameter gradOutput: Gradient from next layer (outputSize x 1)
    /// - Returns: Gradient w.r.t. input (inputSize x 1)
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let input = cachedInput else {
            fatalError("Forward must be called before backward")
        }
        
        // Gradient w.r.t. weights: dW = gradOutput * input^T
        #if canImport(Metal)
        do {
            weightGrad = try Matrix.multiplyGPU(gradOutput, input, transposeB: true)
        } catch {
            weightGrad = Matrix.multiply(gradOutput, input, transposeB: true)
        }
        #else
        weightGrad = Matrix.multiply(gradOutput, input, transposeB: true)
        #endif
        
        // Gradient w.r.t. bias: db = gradOutput
        biasGrad = gradOutput
        
        // Gradient w.r.t. input: dX = W^T * gradOutput
        #if canImport(Metal)
        do {
            return try Matrix.multiplyGPU(weights, gradOutput, transposeA: true)
        } catch {
            return Matrix.multiply(weights, gradOutput, transposeA: true)
        }
        #else
        return Matrix.multiply(weights, gradOutput, transposeA: true)
        #endif
    }
    
    public func parameters() -> [Matrix] {
        return [weights, bias]
    }
    
    public func gradients() -> [Matrix] {
        return [weightGrad, biasGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        // Scale gradients by learning rate
        var scaledWeightGrad = weightGrad
        scaledWeightGrad.scale(by: learningRate)
        
        var scaledBiasGrad = biasGrad
        scaledBiasGrad.scale(by: learningRate)
        
        // Update weights: W = W - lr * dW
        weights = Matrix.subtract(weights, scaledWeightGrad)
        
        // Update bias: b = b - lr * db
        bias = Matrix.subtract(bias, scaledBiasGrad)
        
        // Zero gradients
        weightGrad.zero()
        biasGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        weightGrad.scale(by: scale)
        biasGrad.scale(by: scale)
    }
}

/// ReLU activation layer
public class ReLULayer: Layer {
    /// Cached input from forward pass
    private var cachedInput: Matrix?
    
    public init() {}
    
    public func forward(_ input: Matrix) -> Matrix {
        cachedInput = input
        return Activations.relu(input)
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let input = cachedInput else {
            fatalError("Forward must be called before backward")
        }
        return Activations.reluBackward(input: input, gradOutput: gradOutput)
    }
    
    public func parameters() -> [Matrix] {
        return []
    }
    
    public func gradients() -> [Matrix] {
        return []
    }
    
    public func updateParameters(learningRate: Float) {
        // No parameters to update
    }
    
    public func scaleGradients(by scale: Float) {
        // No gradients to scale
    }
}

/// Softmax activation layer
public class SoftmaxLayer: Layer {
    /// Cached output from forward pass
    private var cachedOutput: Matrix?
    
    public init() {}
    
    public func forward(_ input: Matrix) -> Matrix {
        let output = Activations.softmax(input)
        cachedOutput = output
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let output = cachedOutput else {
            fatalError("Forward must be called before backward")
        }
        return Activations.softmaxBackward(output: output, gradOutput: gradOutput)
    }
    
    public func parameters() -> [Matrix] {
        return []
    }
    
    public func gradients() -> [Matrix] {
        return []
    }
    
    public func updateParameters(learningRate: Float) {
        // No parameters to update
    }
    
    public func scaleGradients(by scale: Float) {
        // No gradients to scale
    }
}
