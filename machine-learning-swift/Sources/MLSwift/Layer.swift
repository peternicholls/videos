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

/// Sigmoid activation layer
public class SigmoidLayer: Layer {
    /// Cached output from forward pass
    private var cachedOutput: Matrix?
    
    public init() {}
    
    public func forward(_ input: Matrix) -> Matrix {
        let output = Activations.sigmoid(input)
        cachedOutput = output
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let output = cachedOutput else {
            fatalError("Forward must be called before backward")
        }
        return Activations.sigmoidBackward(output: output, gradOutput: gradOutput)
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

/// Tanh activation layer
public class TanhLayer: Layer {
    /// Cached output from forward pass
    private var cachedOutput: Matrix?
    
    public init() {}
    
    public func forward(_ input: Matrix) -> Matrix {
        let output = Activations.tanh(input)
        cachedOutput = output
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let output = cachedOutput else {
            fatalError("Forward must be called before backward")
        }
        return Activations.tanhBackward(output: output, gradOutput: gradOutput)
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

/// Dropout regularization layer
/// Randomly sets a fraction of inputs to zero during training
public class DropoutLayer: Layer {
    /// Dropout probability (fraction of inputs to drop)
    private let dropoutRate: Float
    
    /// Mask used during forward pass
    private var mask: Matrix?
    
    /// Whether the layer is in training mode
    public var training: Bool = true
    
    /// Initialize dropout layer
    /// - Parameter dropoutRate: Fraction of inputs to drop (0.0 to 1.0)
    public init(dropoutRate: Float = 0.5) {
        precondition(dropoutRate >= 0.0 && dropoutRate <= 1.0,
                    "Dropout rate must be between 0.0 and 1.0")
        self.dropoutRate = dropoutRate
    }
    
    public func forward(_ input: Matrix) -> Matrix {
        if !training || dropoutRate == 0.0 {
            // During inference or if dropout is disabled, pass through
            return input
        }
        
        // Create binary mask: 1 with probability (1 - dropoutRate), 0 otherwise
        var maskData = [Float](repeating: 0.0, count: input.count)
        for i in 0..<maskData.count {
            maskData[i] = Float.random(in: 0..<1) > dropoutRate ? 1.0 : 0.0
        }
        
        let mask = Matrix(rows: input.rows, cols: input.cols, data: maskData)
        self.mask = mask
        
        // Apply mask and scale by (1 / (1 - dropoutRate)) to maintain expected value
        var output = input
        let scale = 1.0 / (1.0 - dropoutRate)
        for i in 0..<output.data.count {
            output.data[i] *= mask.data[i] * scale
        }
        
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let mask = mask else {
            // If no mask (inference mode), pass gradient through
            return gradOutput
        }
        
        // Apply same mask to gradients
        var gradInput = gradOutput
        let scale = 1.0 / (1.0 - dropoutRate)
        for i in 0..<gradInput.data.count {
            gradInput.data[i] *= mask.data[i] * scale
        }
        
        return gradInput
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

/// Batch Normalization layer
/// Normalizes inputs across the batch dimension
public class BatchNormLayer: Layer {
    /// Number of features to normalize
    private let numFeatures: Int
    
    /// Learnable scale parameter (gamma)
    public var gamma: Matrix
    
    /// Learnable shift parameter (beta)
    public var beta: Matrix
    
    /// Running mean for inference
    private var runningMean: Matrix
    
    /// Running variance for inference
    private var runningVar: Matrix
    
    /// Momentum for running statistics
    private let momentum: Float
    
    /// Small constant for numerical stability
    private let epsilon: Float
    
    /// Whether the layer is in training mode
    public var training: Bool = true
    
    /// Cached values for backward pass
    private var cachedInput: Matrix?
    private var cachedMean: Matrix?
    private var cachedVar: Matrix?
    private var cachedNormalized: Matrix?
    
    /// Gradients
    private var gammaGrad: Matrix
    private var betaGrad: Matrix
    
    /// Initialize batch normalization layer
    /// - Parameters:
    ///   - numFeatures: Number of features to normalize
    ///   - momentum: Momentum for running statistics (default 0.9)
    ///   - epsilon: Small constant for numerical stability (default 1e-5)
    public init(numFeatures: Int, momentum: Float = 0.9, epsilon: Float = 1e-5) {
        self.numFeatures = numFeatures
        self.momentum = momentum
        self.epsilon = epsilon
        
        // Initialize gamma to 1, beta to 0
        self.gamma = Matrix(rows: numFeatures, cols: 1, value: 1.0)
        self.beta = Matrix(rows: numFeatures, cols: 1, value: 0.0)
        
        // Initialize running statistics
        self.runningMean = Matrix(rows: numFeatures, cols: 1, value: 0.0)
        self.runningVar = Matrix(rows: numFeatures, cols: 1, value: 1.0)
        
        // Initialize gradients
        self.gammaGrad = Matrix(rows: numFeatures, cols: 1)
        self.betaGrad = Matrix(rows: numFeatures, cols: 1)
    }
    
    public func forward(_ input: Matrix) -> Matrix {
        precondition(input.rows == numFeatures, "Input features must match numFeatures")
        
        if training {
            // Training mode: compute batch statistics
            cachedInput = input
            
            // For single sample (cols=1), use the sample itself as mean
            let mean = Matrix(rows: numFeatures, cols: 1, value: 0.0)
            
            // Compute mean
            var meanData = mean.data
            for i in 0..<numFeatures {
                meanData[i] = input.data[i]
            }
            let meanMatrix = Matrix(rows: numFeatures, cols: 1, data: meanData)
            cachedMean = meanMatrix
            
            // For single sample, variance is 0, so we use epsilon
            cachedVar = Matrix(rows: numFeatures, cols: 1, value: epsilon)
            
            // Normalize
            var normalized = Matrix(rows: numFeatures, cols: 1)
            for i in 0..<numFeatures {
                normalized.data[i] = (input.data[i] - meanMatrix.data[i]) / sqrt(cachedVar!.data[i] + epsilon)
            }
            cachedNormalized = normalized
            
            // Scale and shift
            var output = Matrix(rows: numFeatures, cols: 1)
            for i in 0..<numFeatures {
                output.data[i] = gamma.data[i] * normalized.data[i] + beta.data[i]
            }
            
            // Update running statistics
            for i in 0..<numFeatures {
                runningMean.data[i] = momentum * runningMean.data[i] + (1.0 - momentum) * meanMatrix.data[i]
                runningVar.data[i] = momentum * runningVar.data[i] + (1.0 - momentum) * cachedVar!.data[i]
            }
            
            return output
        } else {
            // Inference mode: use running statistics
            var normalized = Matrix(rows: numFeatures, cols: 1)
            for i in 0..<numFeatures {
                normalized.data[i] = (input.data[i] - runningMean.data[i]) / sqrt(runningVar.data[i] + epsilon)
            }
            
            var output = Matrix(rows: numFeatures, cols: 1)
            for i in 0..<numFeatures {
                output.data[i] = gamma.data[i] * normalized.data[i] + beta.data[i]
            }
            
            return output
        }
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let _ = cachedInput,
              let normalized = cachedNormalized,
              let variance = cachedVar else {
            fatalError("Forward must be called before backward")
        }
        
        // Compute gradients w.r.t. gamma and beta
        gammaGrad = Matrix(rows: numFeatures, cols: 1)
        betaGrad = Matrix(rows: numFeatures, cols: 1)
        
        for i in 0..<numFeatures {
            gammaGrad.data[i] = gradOutput.data[i] * normalized.data[i]
            betaGrad.data[i] = gradOutput.data[i]
        }
        
        // Compute gradient w.r.t. normalized input
        var gradNormalized = Matrix(rows: numFeatures, cols: 1)
        for i in 0..<numFeatures {
            gradNormalized.data[i] = gradOutput.data[i] * gamma.data[i]
        }
        
        // Compute gradient w.r.t. input
        // For single sample, simplified gradient computation
        var gradInput = Matrix(rows: numFeatures, cols: 1)
        for i in 0..<numFeatures {
            gradInput.data[i] = gradNormalized.data[i] / sqrt(variance.data[i] + epsilon)
        }
        
        return gradInput
    }
    
    public func parameters() -> [Matrix] {
        return [gamma, beta]
    }
    
    public func gradients() -> [Matrix] {
        return [gammaGrad, betaGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        // Scale gradients by learning rate
        var scaledGammaGrad = gammaGrad
        scaledGammaGrad.scale(by: learningRate)
        
        var scaledBetaGrad = betaGrad
        scaledBetaGrad.scale(by: learningRate)
        
        // Update parameters
        gamma = Matrix.subtract(gamma, scaledGammaGrad)
        beta = Matrix.subtract(beta, scaledBetaGrad)
        
        // Zero gradients
        gammaGrad.zero()
        betaGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        gammaGrad.scale(by: scale)
        betaGrad.scale(by: scale)
    }
}
