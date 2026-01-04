/// Activations.swift
/// Neural network activation functions with Metal GPU acceleration
/// Optimized for Apple Silicon

import Foundation

/// Namespace for activation functions
public enum Activations {
    
    // MARK: - ReLU (Rectified Linear Unit)
    
    /// Apply ReLU activation: output = max(0, input)
    /// Uses GPU acceleration via Metal when available
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix with ReLU applied
    public static func relu(_ input: Matrix) -> Matrix {
        #if canImport(Metal)
        do {
            return try Matrix.reluGPU(input)
        } catch {
            print("Warning: GPU ReLU failed, falling back to CPU: \(error)")
            return reluCPU(input)
        }
        #else
        return reluCPU(input)
        #endif
    }
    
    /// CPU-based ReLU implementation (fallback)
    private static func reluCPU(_ input: Matrix) -> Matrix {
        var output = input
        for i in 0..<output.data.count {
            output.data[i] = max(0.0, output.data[i])
        }
        return output
    }
    
    /// ReLU backward pass: compute gradient
    /// grad_input = (input > 0) ? grad_output : 0
    /// - Parameters:
    ///   - input: Original input to forward pass
    ///   - gradOutput: Gradient from next layer
    /// - Returns: Gradient w.r.t. input
    public static func reluBackward(input: Matrix, gradOutput: Matrix) -> Matrix {
        precondition(input.rows == gradOutput.rows && input.cols == gradOutput.cols,
                    "Input and gradient dimensions must match")
        
        var gradInput = Matrix(rows: input.rows, cols: input.cols)
        for i in 0..<input.data.count {
            gradInput.data[i] = input.data[i] > 0.0 ? gradOutput.data[i] : 0.0
        }
        return gradInput
    }
    
    // MARK: - Softmax
    
    /// Apply softmax activation: output[i] = exp(input[i]) / sum(exp(input))
    /// Uses GPU acceleration via Metal when available
    /// - Parameter input: Input matrix (vector)
    /// - Returns: Output matrix with softmax applied
    public static func softmax(_ input: Matrix) -> Matrix {
        #if canImport(Metal)
        do {
            return try Matrix.softmaxGPU(input)
        } catch {
            print("Warning: GPU Softmax failed, falling back to CPU: \(error)")
            return softmaxCPU(input)
        }
        #else
        return softmaxCPU(input)
        #endif
    }
    
    /// CPU-based softmax implementation (fallback)
    private static func softmaxCPU(_ input: Matrix) -> Matrix {
        var output = input
        
        // Find max for numerical stability
        let maxVal = input.data.max() ?? 0.0
        
        // Compute exp and sum
        var sum: Float = 0.0
        for i in 0..<output.data.count {
            output.data[i] = exp(output.data[i] - maxVal)
            sum += output.data[i]
        }
        
        // Normalize
        for i in 0..<output.data.count {
            output.data[i] /= sum
        }
        
        return output
    }
    
    /// Softmax backward pass: compute Jacobian-vector product
    /// For softmax output y and upstream gradient g:
    /// grad_input[i] = sum_j(jacobian[i,j] * g[j])
    /// where jacobian[i,j] = y[i] * (δ[i,j] - y[j])
    /// - Parameters:
    ///   - output: Softmax output from forward pass
    ///   - gradOutput: Gradient from next layer
    /// - Returns: Gradient w.r.t. input
    public static func softmaxBackward(output: Matrix, gradOutput: Matrix) -> Matrix {
        precondition(output.rows == gradOutput.rows && output.cols == gradOutput.cols,
                    "Output and gradient dimensions must match")
        precondition(output.rows == 1 || output.cols == 1,
                    "Softmax currently only supports vectors")
        
        let size = max(output.rows, output.cols)
        var gradInput = Matrix(rows: output.rows, cols: output.cols)
        
        // Compute Jacobian-vector product
        for i in 0..<size {
            var sum: Float = 0.0
            for j in 0..<size {
                let jacobian = output.data[i] * ((i == j ? 1.0 : 0.0) - output.data[j])
                sum += jacobian * gradOutput.data[j]
            }
            gradInput.data[i] = sum
        }
        
        return gradInput
    }
    
    // MARK: - Sigmoid (for future use)
    
    /// Apply sigmoid activation: output = 1 / (1 + exp(-input))
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix with sigmoid applied
    public static func sigmoid(_ input: Matrix) -> Matrix {
        var output = input
        for i in 0..<output.data.count {
            output.data[i] = 1.0 / (1.0 + exp(-output.data[i]))
        }
        return output
    }
    
    /// Sigmoid backward pass
    /// grad_input = output * (1 - output) * grad_output
    /// - Parameters:
    ///   - output: Sigmoid output from forward pass
    ///   - gradOutput: Gradient from next layer
    /// - Returns: Gradient w.r.t. input
    public static func sigmoidBackward(output: Matrix, gradOutput: Matrix) -> Matrix {
        precondition(output.rows == gradOutput.rows && output.cols == gradOutput.cols,
                    "Output and gradient dimensions must match")
        
        var gradInput = Matrix(rows: output.rows, cols: output.cols)
        for i in 0..<output.data.count {
            gradInput.data[i] = output.data[i] * (1.0 - output.data[i]) * gradOutput.data[i]
        }
        return gradInput
    }
    
    // MARK: - Tanh (for future use)
    
    /// Apply tanh activation: output = tanh(input)
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix with tanh applied
    public static func tanh(_ input: Matrix) -> Matrix {
        var output = input
        for i in 0..<output.data.count {
            output.data[i] = Foundation.tanh(output.data[i])
        }
        return output
    }
    
    /// Tanh backward pass
    /// grad_input = (1 - output^2) * grad_output
    /// - Parameters:
    ///   - output: Tanh output from forward pass
    ///   - gradOutput: Gradient from next layer
    /// - Returns: Gradient w.r.t. input
    public static func tanhBackward(output: Matrix, gradOutput: Matrix) -> Matrix {
        precondition(output.rows == gradOutput.rows && output.cols == gradOutput.cols,
                    "Output and gradient dimensions must match")
        
        var gradInput = Matrix(rows: output.rows, cols: output.cols)
        for i in 0..<output.data.count {
            gradInput.data[i] = (1.0 - output.data[i] * output.data[i]) * gradOutput.data[i]
        }
        return gradInput
    }
}
