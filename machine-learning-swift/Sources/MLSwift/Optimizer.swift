/// Optimizer.swift
/// Optimization algorithms for neural network training
/// Optimized for Apple Silicon

import Foundation

/// Protocol for optimization algorithms
public protocol Optimizer {
    /// Update parameters using gradients
    /// - Parameters:
    ///   - parameters: Array of parameter matrices to update
    ///   - gradients: Array of gradient matrices
    ///   - learningRate: Learning rate for the update
    func update(parameters: inout [Matrix], gradients: [Matrix], learningRate: Float)
    
    /// Reset optimizer state (for a new training run)
    func reset()
}

/// Stochastic Gradient Descent (SGD) optimizer
public class SGDOptimizer: Optimizer {
    public init() {}
    
    public func update(parameters: inout [Matrix], gradients: [Matrix], learningRate: Float) {
        precondition(parameters.count == gradients.count,
                    "Parameters and gradients count must match")
        
        for i in 0..<parameters.count {
            var scaledGrad = gradients[i]
            scaledGrad.scale(by: learningRate)
            parameters[i] = Matrix.subtract(parameters[i], scaledGrad)
        }
    }
    
    public func reset() {
        // SGD has no state to reset
    }
}

/// SGD with Momentum optimizer
public class SGDMomentumOptimizer: Optimizer {
    /// Momentum coefficient (typically 0.9)
    private let momentum: Float
    
    /// Velocity terms for each parameter
    private var velocities: [Matrix]
    
    /// Initialize SGD with momentum
    /// - Parameter momentum: Momentum coefficient (default 0.9)
    public init(momentum: Float = 0.9) {
        precondition(momentum >= 0.0 && momentum < 1.0,
                    "Momentum must be between 0.0 and 1.0")
        self.momentum = momentum
        self.velocities = []
    }
    
    public func update(parameters: inout [Matrix], gradients: [Matrix], learningRate: Float) {
        precondition(parameters.count == gradients.count,
                    "Parameters and gradients count must match")
        
        // Initialize velocities on first call
        if velocities.isEmpty {
            velocities = parameters.map { Matrix(rows: $0.rows, cols: $0.cols) }
        }
        
        for i in 0..<parameters.count {
            // v = momentum * v + learning_rate * gradient
            velocities[i].scale(by: momentum)
            var scaledGrad = gradients[i]
            scaledGrad.scale(by: learningRate)
            velocities[i] = Matrix.add(velocities[i], scaledGrad)
            
            // parameter = parameter - v
            parameters[i] = Matrix.subtract(parameters[i], velocities[i])
        }
    }
    
    public func reset() {
        velocities.removeAll()
    }
}

/// Adam (Adaptive Moment Estimation) optimizer
public class AdamOptimizer: Optimizer {
    /// Learning rate decay factor for first moment (typically 0.9)
    private let beta1: Float
    
    /// Learning rate decay factor for second moment (typically 0.999)
    private let beta2: Float
    
    /// Small constant for numerical stability
    private let epsilon: Float
    
    /// First moment estimates (mean of gradients)
    private var m: [Matrix]
    
    /// Second moment estimates (uncentered variance of gradients)
    private var v: [Matrix]
    
    /// Time step counter
    private var t: Int
    
    /// Initialize Adam optimizer
    /// - Parameters:
    ///   - beta1: Decay rate for first moment (default 0.9)
    ///   - beta2: Decay rate for second moment (default 0.999)
    ///   - epsilon: Small constant for numerical stability (default 1e-8)
    public init(beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        precondition(beta1 >= 0.0 && beta1 < 1.0, "beta1 must be between 0.0 and 1.0")
        precondition(beta2 >= 0.0 && beta2 < 1.0, "beta2 must be between 0.0 and 1.0")
        precondition(epsilon > 0.0, "epsilon must be positive")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0
    }
    
    public func update(parameters: inout [Matrix], gradients: [Matrix], learningRate: Float) {
        precondition(parameters.count == gradients.count,
                    "Parameters and gradients count must match")
        
        // Initialize moment estimates on first call
        if m.isEmpty {
            m = parameters.map { Matrix(rows: $0.rows, cols: $0.cols) }
            v = parameters.map { Matrix(rows: $0.rows, cols: $0.cols) }
        }
        
        t += 1
        
        for i in 0..<parameters.count {
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
            m[i].scale(by: beta1)
            var gradScaled = gradients[i]
            gradScaled.scale(by: 1.0 - beta1)
            m[i] = Matrix.add(m[i], gradScaled)
            
            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * gradient^2
            v[i].scale(by: beta2)
            var gradSquared = gradients[i]
            for j in 0..<gradSquared.data.count {
                gradSquared.data[j] = gradSquared.data[j] * gradSquared.data[j] * (1.0 - beta2)
            }
            v[i] = Matrix.add(v[i], gradSquared)
            
            // Compute bias-corrected first moment estimate
            var mHat = m[i]
            let biasCorrection1 = 1.0 - pow(beta1, Float(t))
            mHat.scale(by: 1.0 / biasCorrection1)
            
            // Compute bias-corrected second moment estimate
            var vHat = v[i]
            let biasCorrection2 = 1.0 - pow(beta2, Float(t))
            vHat.scale(by: 1.0 / biasCorrection2)
            
            // Update parameters: parameter = parameter - learningRate * mHat / (sqrt(vHat) + epsilon)
            var update = Matrix(rows: parameters[i].rows, cols: parameters[i].cols)
            for j in 0..<update.data.count {
                update.data[j] = learningRate * mHat.data[j] / (sqrt(vHat.data[j]) + epsilon)
            }
            
            parameters[i] = Matrix.subtract(parameters[i], update)
        }
    }
    
    public func reset() {
        m.removeAll()
        v.removeAll()
        t = 0
    }
}

/// RMSprop optimizer
public class RMSpropOptimizer: Optimizer {
    /// Decay rate for moving average (typically 0.9)
    private let decay: Float
    
    /// Small constant for numerical stability
    private let epsilon: Float
    
    /// Moving average of squared gradients
    private var cache: [Matrix]
    
    /// Initialize RMSprop optimizer
    /// - Parameters:
    ///   - decay: Decay rate for moving average (default 0.9)
    ///   - epsilon: Small constant for numerical stability (default 1e-8)
    public init(decay: Float = 0.9, epsilon: Float = 1e-8) {
        precondition(decay >= 0.0 && decay < 1.0, "decay must be between 0.0 and 1.0")
        precondition(epsilon > 0.0, "epsilon must be positive")
        
        self.decay = decay
        self.epsilon = epsilon
        self.cache = []
    }
    
    public func update(parameters: inout [Matrix], gradients: [Matrix], learningRate: Float) {
        precondition(parameters.count == gradients.count,
                    "Parameters and gradients count must match")
        
        // Initialize cache on first call
        if cache.isEmpty {
            cache = parameters.map { Matrix(rows: $0.rows, cols: $0.cols) }
        }
        
        for i in 0..<parameters.count {
            // Update cache: cache = decay * cache + (1 - decay) * gradient^2
            cache[i].scale(by: decay)
            var gradSquared = gradients[i]
            for j in 0..<gradSquared.data.count {
                gradSquared.data[j] = gradSquared.data[j] * gradSquared.data[j] * (1.0 - decay)
            }
            cache[i] = Matrix.add(cache[i], gradSquared)
            
            // Update parameters: parameter = parameter - learningRate * gradient / (sqrt(cache) + epsilon)
            var update = Matrix(rows: parameters[i].rows, cols: parameters[i].cols)
            for j in 0..<update.data.count {
                update.data[j] = learningRate * gradients[i].data[j] / (sqrt(cache[i].data[j]) + epsilon)
            }
            
            parameters[i] = Matrix.subtract(parameters[i], update)
        }
    }
    
    public func reset() {
        cache.removeAll()
    }
}
