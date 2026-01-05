/// GradientClipping.swift
/// Gradient clipping utilities to prevent exploding gradients
/// Optimized for Apple Silicon

import Foundation

/// Gradient clipping configuration
public enum GradientClipConfig {
    /// Clip gradients by absolute value
    case byValue(maxValue: Float)
    
    /// Clip gradients by per-parameter L2 norm
    case byNorm(maxNorm: Float)
    
    /// Clip gradients by global L2 norm (across all parameters)
    case globalNorm(maxNorm: Float)
}

/// Gradient clipping utilities
public enum GradientClipping {
    
    /// Clip gradients according to the specified configuration
    /// - Parameters:
    ///   - gradients: Array of gradient matrices to clip (modified in place)
    ///   - config: Clipping configuration
    /// - Returns: The global gradient norm before clipping (useful for monitoring)
    @discardableResult
    public static func clip(_ gradients: inout [Matrix], config: GradientClipConfig) -> Float {
        switch config {
        case .byValue(let maxValue):
            return clipByValue(&gradients, maxValue: maxValue)
        case .byNorm(let maxNorm):
            return clipByNorm(&gradients, maxNorm: maxNorm)
        case .globalNorm(let maxNorm):
            return clipByGlobalNorm(&gradients, maxNorm: maxNorm)
        }
    }
    
    // MARK: - Clip by Value
    
    /// Clip gradient values to be within [-maxValue, maxValue]
    /// - Parameters:
    ///   - gradients: Array of gradient matrices to clip
    ///   - maxValue: Maximum absolute value allowed
    /// - Returns: Maximum absolute gradient value before clipping
    @discardableResult
    public static func clipByValue(_ gradients: inout [Matrix], maxValue: Float) -> Float {
        precondition(maxValue > 0, "maxValue must be positive")
        
        var maxAbsValue: Float = 0
        
        for i in 0..<gradients.count {
            for j in 0..<gradients[i].data.count {
                let absVal = abs(gradients[i].data[j])
                maxAbsValue = max(maxAbsValue, absVal)
                
                // Clip to [-maxValue, maxValue]
                gradients[i].data[j] = min(max(gradients[i].data[j], -maxValue), maxValue)
            }
        }
        
        return maxAbsValue
    }
    
    // MARK: - Clip by Per-Parameter Norm
    
    /// Clip each gradient matrix by its L2 norm
    /// - Parameters:
    ///   - gradients: Array of gradient matrices to clip
    ///   - maxNorm: Maximum L2 norm allowed per parameter
    /// - Returns: Maximum L2 norm before clipping
    @discardableResult
    public static func clipByNorm(_ gradients: inout [Matrix], maxNorm: Float) -> Float {
        precondition(maxNorm > 0, "maxNorm must be positive")
        
        var maxNormFound: Float = 0
        
        for i in 0..<gradients.count {
            let norm = l2Norm(gradients[i])
            maxNormFound = max(maxNormFound, norm)
            
            if norm > maxNorm {
                let scale = maxNorm / norm
                gradients[i].scale(by: scale)
            }
        }
        
        return maxNormFound
    }
    
    // MARK: - Clip by Global Norm
    
    /// Clip gradients by global L2 norm (computed across all parameters)
    /// This is the most commonly used clipping strategy for RNNs
    /// - Parameters:
    ///   - gradients: Array of gradient matrices to clip
    ///   - maxNorm: Maximum global L2 norm allowed
    /// - Returns: Global L2 norm before clipping
    @discardableResult
    public static func clipByGlobalNorm(_ gradients: inout [Matrix], maxNorm: Float) -> Float {
        precondition(maxNorm > 0, "maxNorm must be positive")
        
        // Compute global norm
        let globalNorm = computeGlobalNorm(gradients)
        
        // Scale all gradients if norm exceeds threshold
        if globalNorm > maxNorm {
            let scale = maxNorm / globalNorm
            for i in 0..<gradients.count {
                gradients[i].scale(by: scale)
            }
        }
        
        return globalNorm
    }
    
    // MARK: - Utility Functions
    
    /// Compute L2 norm of a matrix
    /// - Parameter matrix: Input matrix
    /// - Returns: L2 norm (Frobenius norm)
    public static func l2Norm(_ matrix: Matrix) -> Float {
        var sumSquares: Float = 0
        for value in matrix.data {
            sumSquares += value * value
        }
        return sqrt(sumSquares)
    }
    
    /// Compute global L2 norm across all gradient matrices
    /// - Parameter gradients: Array of gradient matrices
    /// - Returns: Global L2 norm
    public static func computeGlobalNorm(_ gradients: [Matrix]) -> Float {
        var sumSquares: Float = 0
        for grad in gradients {
            for value in grad.data {
                sumSquares += value * value
            }
        }
        return sqrt(sumSquares)
    }
}

// MARK: - Model Extension for Gradient Clipping

extension SequentialModel {
    /// Train on a single example with gradient clipping
    /// - Parameters:
    ///   - input: Input matrix
    ///   - target: Target output matrix
    ///   - learningRate: Learning rate for gradient descent
    ///   - clipConfig: Gradient clipping configuration
    /// - Returns: Tuple of (loss, gradient norm before clipping)
    @discardableResult
    public func trainStepWithClipping(
        input: Matrix,
        target: Matrix,
        learningRate: Float,
        clipConfig: GradientClipConfig
    ) -> (loss: Float, gradNorm: Float) {
        // Forward pass
        let predicted = forward(input)
        
        // Compute loss (need to access loss function - simplified approach)
        let loss = Loss.crossEntropy(predicted: predicted, target: target)
        
        // Backward pass
        let gradOutput = Loss.crossEntropyBackward(predicted: predicted, target: target)
        backward(gradOutput)
        
        // Collect all gradients
        var allGradients: [Matrix] = []
        for layer in getLayers() {
            allGradients.append(contentsOf: layer.gradients())
        }
        
        // Clip gradients
        let gradNorm = GradientClipping.clip(&allGradients, config: clipConfig)
        
        // Update parameters
        for layer in getLayers() {
            layer.updateParameters(learningRate: learningRate)
        }
        
        return (loss, gradNorm)
    }
}
