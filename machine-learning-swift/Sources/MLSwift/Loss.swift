/// Loss.swift
/// Loss functions for neural network training
/// Optimized for Apple Silicon with Metal GPU acceleration

import Foundation

/// Namespace for loss functions
public enum Loss {
    
    // MARK: - Cross-Entropy Loss
    
    /// Compute cross-entropy loss: loss = -sum(p * log(q))
    /// Where p is the true distribution and q is the predicted distribution
    /// - Parameters:
    ///   - predicted: Predicted probability distribution
    ///   - target: True probability distribution (one-hot encoded labels)
    /// - Returns: Scalar loss value
    public static func crossEntropy(predicted: Matrix, target: Matrix) -> Float {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var loss: Float = 0.0
        let epsilon: Float = 1e-7  // Small constant to avoid log(0)
        
        for i in 0..<predicted.data.count {
            if target.data[i] > 0.0 {
                let q = max(predicted.data[i], epsilon)
                loss -= target.data[i] * log(q)
            }
        }
        
        return loss
    }
    
    /// Cross-entropy backward pass: compute gradients
    /// grad_predicted = -target / predicted
    /// - Parameters:
    ///   - predicted: Predicted probability distribution
    ///   - target: True probability distribution
    /// - Returns: Gradient w.r.t. predicted
    public static func crossEntropyBackward(predicted: Matrix, target: Matrix) -> Matrix {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var grad = Matrix(rows: predicted.rows, cols: predicted.cols)
        let epsilon: Float = 1e-7
        
        for i in 0..<predicted.data.count {
            let q = max(predicted.data[i], epsilon)
            grad.data[i] = -target.data[i] / q
        }
        
        return grad
    }
    
    // MARK: - Mean Squared Error (MSE)
    
    /// Compute mean squared error: loss = mean((predicted - target)^2)
    /// - Parameters:
    ///   - predicted: Predicted values
    ///   - target: True values
    /// - Returns: Scalar loss value
    public static func meanSquaredError(predicted: Matrix, target: Matrix) -> Float {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var sumSquaredError: Float = 0.0
        for i in 0..<predicted.data.count {
            let diff = predicted.data[i] - target.data[i]
            sumSquaredError += diff * diff
        }
        
        return sumSquaredError / Float(predicted.data.count)
    }
    
    /// MSE backward pass: compute gradients
    /// grad_predicted = 2 * (predicted - target) / n
    /// - Parameters:
    ///   - predicted: Predicted values
    ///   - target: True values
    /// - Returns: Gradient w.r.t. predicted
    public static func meanSquaredErrorBackward(predicted: Matrix, target: Matrix) -> Matrix {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var grad = Matrix(rows: predicted.rows, cols: predicted.cols)
        let scale = 2.0 / Float(predicted.data.count)
        
        for i in 0..<predicted.data.count {
            grad.data[i] = scale * (predicted.data[i] - target.data[i])
        }
        
        return grad
    }
    
    // MARK: - Binary Cross-Entropy (for future use)
    
    /// Compute binary cross-entropy loss
    /// loss = -mean(target * log(predicted) + (1 - target) * log(1 - predicted))
    /// - Parameters:
    ///   - predicted: Predicted probabilities in [0, 1]
    ///   - target: True binary labels (0 or 1)
    /// - Returns: Scalar loss value
    public static func binaryCrossEntropy(predicted: Matrix, target: Matrix) -> Float {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var loss: Float = 0.0
        let epsilon: Float = 1e-7
        
        for i in 0..<predicted.data.count {
            let p = max(min(predicted.data[i], 1.0 - epsilon), epsilon)
            let t = target.data[i]
            loss -= t * log(p) + (1.0 - t) * log(1.0 - p)
        }
        
        return loss / Float(predicted.data.count)
    }
    
    /// Binary cross-entropy backward pass
    /// grad_predicted = (predicted - target) / (predicted * (1 - predicted))
    /// - Parameters:
    ///   - predicted: Predicted probabilities
    ///   - target: True binary labels
    /// - Returns: Gradient w.r.t. predicted
    public static func binaryCrossEntropyBackward(predicted: Matrix, target: Matrix) -> Matrix {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols,
                    "Predicted and target dimensions must match")
        
        var grad = Matrix(rows: predicted.rows, cols: predicted.cols)
        let epsilon: Float = 1e-7
        let scale = 1.0 / Float(predicted.data.count)
        
        for i in 0..<predicted.data.count {
            let p = max(min(predicted.data[i], 1.0 - epsilon), epsilon)
            let t = target.data[i]
            grad.data[i] = scale * (p - t) / (p * (1.0 - p))
        }
        
        return grad
    }
}
