/// LearningRateScheduler.swift
/// Learning rate scheduling strategies for neural network training
/// Optimized for Apple Silicon

import Foundation

/// Protocol for learning rate schedulers
public protocol LearningRateSchedulerProtocol {
    /// Get the learning rate for a given epoch
    /// - Parameter epoch: Current epoch (1-indexed)
    /// - Returns: Learning rate for this epoch
    func learningRate(for epoch: Int) -> Float
    
    /// Reset the scheduler state
    func reset()
}

/// Learning rate scheduler factory and implementations
public enum LearningRateScheduler {
    
    // MARK: - Step Decay Scheduler
    
    /// Creates a step decay scheduler that reduces learning rate by a factor every N epochs
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - decayFactor: Factor to multiply learning rate by (typically 0.1)
    ///   - decayEvery: Number of epochs between decays
    /// - Returns: A step decay scheduler
    public static func stepDecay(
        initialLR: Float,
        decayFactor: Float,
        decayEvery: Int
    ) -> StepDecayScheduler {
        return StepDecayScheduler(
            initialLR: initialLR,
            decayFactor: decayFactor,
            decayEvery: decayEvery
        )
    }
    
    /// Creates an exponential decay scheduler
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - decayRate: Exponential decay rate (0 < decayRate < 1)
    /// - Returns: An exponential decay scheduler
    public static func exponentialDecay(
        initialLR: Float,
        decayRate: Float
    ) -> ExponentialDecayScheduler {
        return ExponentialDecayScheduler(
            initialLR: initialLR,
            decayRate: decayRate
        )
    }
    
    /// Creates a cosine annealing scheduler
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - minLR: Minimum learning rate
    ///   - cycleLength: Number of epochs per cycle
    /// - Returns: A cosine annealing scheduler
    public static func cosineAnnealing(
        initialLR: Float,
        minLR: Float,
        cycleLength: Int
    ) -> CosineAnnealingScheduler {
        return CosineAnnealingScheduler(
            initialLR: initialLR,
            minLR: minLR,
            cycleLength: cycleLength
        )
    }
    
    /// Creates a warmup scheduler that linearly increases LR for N epochs then uses another scheduler
    /// - Parameters:
    ///   - warmupEpochs: Number of warmup epochs
    ///   - initialLR: Learning rate to start warmup from
    ///   - targetLR: Learning rate to reach after warmup
    ///   - afterWarmup: Scheduler to use after warmup (optional)
    /// - Returns: A warmup scheduler
    public static func warmup(
        warmupEpochs: Int,
        initialLR: Float,
        targetLR: Float,
        afterWarmup: LearningRateSchedulerProtocol? = nil
    ) -> WarmupScheduler {
        return WarmupScheduler(
            warmupEpochs: warmupEpochs,
            initialLR: initialLR,
            targetLR: targetLR,
            afterWarmup: afterWarmup
        )
    }
    
    /// Creates a reduce on plateau scheduler
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - factor: Factor to reduce learning rate by
    ///   - patience: Number of epochs to wait before reducing
    ///   - minLR: Minimum learning rate
    /// - Returns: A reduce on plateau scheduler
    public static func reduceOnPlateau(
        initialLR: Float,
        factor: Float = 0.1,
        patience: Int = 10,
        minLR: Float = 1e-6
    ) -> ReduceOnPlateauScheduler {
        return ReduceOnPlateauScheduler(
            initialLR: initialLR,
            factor: factor,
            patience: patience,
            minLR: minLR
        )
    }
}

// MARK: - Step Decay Scheduler Implementation

/// Reduces learning rate by a factor every N epochs
public class StepDecayScheduler: LearningRateSchedulerProtocol {
    private let initialLR: Float
    private let decayFactor: Float
    private let decayEvery: Int
    
    public init(initialLR: Float, decayFactor: Float, decayEvery: Int) {
        precondition(initialLR > 0, "Initial learning rate must be positive")
        precondition(decayFactor > 0 && decayFactor <= 1, "Decay factor must be in (0, 1]")
        precondition(decayEvery > 0, "Decay interval must be positive")
        
        self.initialLR = initialLR
        self.decayFactor = decayFactor
        self.decayEvery = decayEvery
    }
    
    public func learningRate(for epoch: Int) -> Float {
        let numDecays = (epoch - 1) / decayEvery
        return initialLR * pow(decayFactor, Float(numDecays))
    }
    
    public func reset() {
        // Stateless scheduler, nothing to reset
    }
}

// MARK: - Exponential Decay Scheduler Implementation

/// Exponentially decays learning rate each epoch
public class ExponentialDecayScheduler: LearningRateSchedulerProtocol {
    private let initialLR: Float
    private let decayRate: Float
    
    public init(initialLR: Float, decayRate: Float) {
        precondition(initialLR > 0, "Initial learning rate must be positive")
        precondition(decayRate > 0 && decayRate < 1, "Decay rate must be in (0, 1)")
        
        self.initialLR = initialLR
        self.decayRate = decayRate
    }
    
    public func learningRate(for epoch: Int) -> Float {
        return initialLR * pow(decayRate, Float(epoch - 1))
    }
    
    public func reset() {
        // Stateless scheduler, nothing to reset
    }
}

// MARK: - Cosine Annealing Scheduler Implementation

/// Cosine annealing learning rate schedule
public class CosineAnnealingScheduler: LearningRateSchedulerProtocol {
    private let initialLR: Float
    private let minLR: Float
    private let cycleLength: Int
    
    public init(initialLR: Float, minLR: Float, cycleLength: Int) {
        precondition(initialLR > 0, "Initial learning rate must be positive")
        precondition(minLR >= 0 && minLR < initialLR, "Min LR must be non-negative and less than initial LR")
        precondition(cycleLength > 0, "Cycle length must be positive")
        
        self.initialLR = initialLR
        self.minLR = minLR
        self.cycleLength = cycleLength
    }
    
    public func learningRate(for epoch: Int) -> Float {
        // Position within current cycle (0 to 1)
        let cyclePosition = Float((epoch - 1) % cycleLength) / Float(cycleLength)
        
        // Cosine annealing formula
        let cosValue = (1.0 + cos(Float.pi * cyclePosition)) / 2.0
        return minLR + (initialLR - minLR) * cosValue
    }
    
    public func reset() {
        // Stateless scheduler, nothing to reset
    }
}

// MARK: - Warmup Scheduler Implementation

/// Linearly warms up learning rate then optionally delegates to another scheduler
public class WarmupScheduler: LearningRateSchedulerProtocol {
    private let warmupEpochs: Int
    private let initialLR: Float
    private let targetLR: Float
    private let afterWarmup: LearningRateSchedulerProtocol?
    
    public init(
        warmupEpochs: Int,
        initialLR: Float,
        targetLR: Float,
        afterWarmup: LearningRateSchedulerProtocol? = nil
    ) {
        precondition(warmupEpochs > 0, "Warmup epochs must be positive")
        precondition(initialLR >= 0, "Initial learning rate must be non-negative")
        precondition(targetLR > 0, "Target learning rate must be positive")
        
        self.warmupEpochs = warmupEpochs
        self.initialLR = initialLR
        self.targetLR = targetLR
        self.afterWarmup = afterWarmup
    }
    
    public func learningRate(for epoch: Int) -> Float {
        if epoch <= warmupEpochs {
            // Linear warmup
            let progress = Float(epoch) / Float(warmupEpochs)
            return initialLR + (targetLR - initialLR) * progress
        } else {
            // After warmup
            if let scheduler = afterWarmup {
                return scheduler.learningRate(for: epoch - warmupEpochs)
            } else {
                return targetLR
            }
        }
    }
    
    public func reset() {
        afterWarmup?.reset()
    }
}

// MARK: - Reduce On Plateau Scheduler Implementation

/// Reduces learning rate when a metric has stopped improving
public class ReduceOnPlateauScheduler: LearningRateSchedulerProtocol {
    private let initialLR: Float
    private let factor: Float
    private let patience: Int
    private let minLR: Float
    
    private var currentLR: Float
    private var bestMetric: Float
    private var epochsWithoutImprovement: Int
    
    public init(
        initialLR: Float,
        factor: Float = 0.1,
        patience: Int = 10,
        minLR: Float = 1e-6
    ) {
        precondition(initialLR > 0, "Initial learning rate must be positive")
        precondition(factor > 0 && factor < 1, "Factor must be in (0, 1)")
        precondition(patience > 0, "Patience must be positive")
        precondition(minLR >= 0, "Min LR must be non-negative")
        
        self.initialLR = initialLR
        self.factor = factor
        self.patience = patience
        self.minLR = minLR
        
        self.currentLR = initialLR
        self.bestMetric = Float.infinity
        self.epochsWithoutImprovement = 0
    }
    
    public func learningRate(for epoch: Int) -> Float {
        return currentLR
    }
    
    /// Update the scheduler with the current metric value
    /// - Parameter metric: Current metric value (lower is better, e.g., loss)
    public func step(metric: Float) {
        if metric < bestMetric {
            bestMetric = metric
            epochsWithoutImprovement = 0
        } else {
            epochsWithoutImprovement += 1
            
            if epochsWithoutImprovement >= patience {
                currentLR = max(currentLR * factor, minLR)
                epochsWithoutImprovement = 0
            }
        }
    }
    
    public func reset() {
        currentLR = initialLR
        bestMetric = Float.infinity
        epochsWithoutImprovement = 0
    }
}

// MARK: - Cyclic Learning Rate Scheduler

/// Cyclic learning rate that oscillates between bounds
public class CyclicLRScheduler: LearningRateSchedulerProtocol {
    private let baseLR: Float
    private let maxLR: Float
    private let stepSize: Int
    private let mode: CyclicMode
    
    public enum CyclicMode {
        case triangular
        case triangular2
        case expRange(gamma: Float)
    }
    
    public init(baseLR: Float, maxLR: Float, stepSize: Int, mode: CyclicMode = .triangular) {
        precondition(baseLR > 0, "Base learning rate must be positive")
        precondition(maxLR > baseLR, "Max LR must be greater than base LR")
        precondition(stepSize > 0, "Step size must be positive")
        
        self.baseLR = baseLR
        self.maxLR = maxLR
        self.stepSize = stepSize
        self.mode = mode
    }
    
    public func learningRate(for epoch: Int) -> Float {
        let cycle = Float(1 + (epoch - 1) / (2 * stepSize))
        let x = Float(abs(epoch - 1 - 2 * Int(cycle) * stepSize + stepSize)) / Float(stepSize)
        
        let scale: Float
        switch mode {
        case .triangular:
            scale = 1.0
        case .triangular2:
            scale = 1.0 / pow(2.0, cycle - 1)
        case .expRange(let gamma):
            scale = pow(gamma, Float(epoch - 1))
        }
        
        return baseLR + (maxLR - baseLR) * max(0, 1 - x) * scale
    }
    
    public func reset() {
        // Stateless scheduler, nothing to reset
    }
}
