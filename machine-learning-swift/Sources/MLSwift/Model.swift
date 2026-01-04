/// Model.swift
/// Sequential neural network model
/// Optimized for Apple Silicon with Metal acceleration

import Foundation

/// Sequential neural network model
public class SequentialModel {
    /// Array of layers in the model
    private var layers: [Layer]
    
    /// Loss function to use for training
    private var lossFunction: (Matrix, Matrix) -> Float
    
    /// Loss gradient function
    private var lossGradient: (Matrix, Matrix) -> Matrix
    
    /// Initialize an empty sequential model
    public init() {
        self.layers = []
        // Default to cross-entropy loss
        self.lossFunction = Loss.crossEntropy
        self.lossGradient = Loss.crossEntropyBackward
    }
    
    /// Add a layer to the model
    /// - Parameter layer: Layer to add
    public func add(_ layer: Layer) {
        layers.append(layer)
    }
    
    /// Set the loss function
    /// - Parameters:
    ///   - loss: Loss function (predicted, target) -> scalar
    ///   - gradient: Loss gradient function (predicted, target) -> gradient
    public func setLoss(
        _ loss: @escaping (Matrix, Matrix) -> Float,
        gradient: @escaping (Matrix, Matrix) -> Matrix
    ) {
        self.lossFunction = loss
        self.lossGradient = gradient
    }
    
    /// Forward pass through the model
    /// - Parameter input: Input matrix
    /// - Returns: Output matrix
    public func forward(_ input: Matrix) -> Matrix {
        var output = input
        for layer in layers {
            output = layer.forward(output)
        }
        return output
    }
    
    /// Backward pass through the model
    /// - Parameter gradOutput: Gradient from loss function
    /// - Returns: Gradient w.r.t. input
    @discardableResult
    public func backward(_ gradOutput: Matrix) -> Matrix {
        var grad = gradOutput
        for layer in layers.reversed() {
            grad = layer.backward(grad)
        }
        return grad
    }
    
    /// Compute loss for a single example
    /// - Parameters:
    ///   - input: Input matrix
    ///   - target: Target output matrix
    /// - Returns: Loss value
    public func computeLoss(input: Matrix, target: Matrix) -> Float {
        let predicted = forward(input)
        return lossFunction(predicted, target)
    }
    
    /// Train on a single example (single gradient descent step)
    /// - Parameters:
    ///   - input: Input matrix
    ///   - target: Target output matrix
    ///   - learningRate: Learning rate for gradient descent
    /// - Returns: Loss value
    @discardableResult
    public func trainStep(input: Matrix, target: Matrix, learningRate: Float) -> Float {
        // Forward pass
        let predicted = forward(input)
        
        // Compute loss
        let loss = lossFunction(predicted, target)
        
        // Backward pass
        let gradOutput = lossGradient(predicted, target)
        backward(gradOutput)
        
        // Update parameters
        for layer in layers {
            layer.updateParameters(learningRate: learningRate)
        }
        
        return loss
    }
    
    /// Train on a batch of examples
    /// - Parameters:
    ///   - inputs: Array of input matrices
    ///   - targets: Array of target matrices
    ///   - learningRate: Learning rate for gradient descent
    /// - Returns: Average loss over the batch
    @discardableResult
    public func trainBatch(
        inputs: [Matrix],
        targets: [Matrix],
        learningRate: Float
    ) -> Float {
        precondition(inputs.count == targets.count, "Input and target counts must match")
        precondition(!inputs.isEmpty, "Batch cannot be empty")
        
        var totalLoss: Float = 0.0
        
        // Accumulate gradients over the batch
        for (input, target) in zip(inputs, targets) {
            // Forward pass
            let predicted = forward(input)
            
            // Compute loss
            totalLoss += lossFunction(predicted, target)
            
            // Backward pass (accumulates gradients)
            let gradOutput = lossGradient(predicted, target)
            backward(gradOutput)
        }
        
        // Average loss
        let avgLoss = totalLoss / Float(inputs.count)
        
        // Scale gradients to average them across the batch
        for layer in layers {
            layer.scaleGradients(by: 1.0 / Float(inputs.count))
        }
        
        // Update parameters with averaged gradients
        for layer in layers {
            layer.updateParameters(learningRate: learningRate)
        }
        
        return avgLoss
    }
    
    /// Evaluate the model on test data
    /// - Parameters:
    ///   - inputs: Array of test inputs
    ///   - targets: Array of test targets
    /// - Returns: Tuple of (average loss, accuracy)
    public func evaluate(inputs: [Matrix], targets: [Matrix]) -> (loss: Float, accuracy: Float) {
        precondition(inputs.count == targets.count, "Input and target counts must match")
        
        var totalLoss: Float = 0.0
        var correctPredictions = 0
        
        for (input, target) in zip(inputs, targets) {
            let predicted = forward(input)
            totalLoss += lossFunction(predicted, target)
            
            // Check if prediction is correct (for classification)
            if predicted.argmax() == target.argmax() {
                correctPredictions += 1
            }
        }
        
        let avgLoss = totalLoss / Float(inputs.count)
        let accuracy = Float(correctPredictions) / Float(inputs.count)
        
        return (avgLoss, accuracy)
    }
    
    /// Train the model for multiple epochs
    /// - Parameters:
    ///   - trainInputs: Training input data
    ///   - trainTargets: Training target data
    ///   - testInputs: Test input data (optional)
    ///   - testTargets: Test target data (optional)
    ///   - epochs: Number of training epochs
    ///   - batchSize: Mini-batch size
    ///   - learningRate: Learning rate
    public func train(
        trainInputs: [Matrix],
        trainTargets: [Matrix],
        testInputs: [Matrix]? = nil,
        testTargets: [Matrix]? = nil,
        epochs: Int,
        batchSize: Int,
        learningRate: Float
    ) {
        precondition(trainInputs.count == trainTargets.count,
                    "Train input and target counts must match")
        
        let numBatches = (trainInputs.count + batchSize - 1) / batchSize
        
        for epoch in 1...epochs {
            // Shuffle training data
            let shuffledIndices = (0..<trainInputs.count).shuffled()
            
            var epochLoss: Float = 0.0
            
            // Train on batches
            for batch in 0..<numBatches {
                let startIdx = batch * batchSize
                let endIdx = min(startIdx + batchSize, trainInputs.count)
                
                var batchInputs: [Matrix] = []
                var batchTargets: [Matrix] = []
                
                for i in startIdx..<endIdx {
                    let idx = shuffledIndices[i]
                    batchInputs.append(trainInputs[idx])
                    batchTargets.append(trainTargets[idx])
                }
                
                let batchLoss = trainBatch(
                    inputs: batchInputs,
                    targets: batchTargets,
                    learningRate: learningRate
                )
                epochLoss += batchLoss
                
                // Print progress (Swift handles flushing automatically)
                print("Epoch \(epoch)/\(epochs), Batch \(batch+1)/\(numBatches), Loss: \(String(format: "%.4f", batchLoss))", terminator: "\r")
            }
            
            let avgEpochLoss = epochLoss / Float(numBatches)
            print("") // New line after progress
            
            // Evaluate on test set if provided
            if let testInputs = testInputs, let testTargets = testTargets {
                let (testLoss, testAccuracy) = evaluate(inputs: testInputs, targets: testTargets)
                print("Epoch \(epoch): Train Loss = \(String(format: "%.4f", avgEpochLoss)), Test Loss = \(String(format: "%.4f", testLoss)), Test Accuracy = \(String(format: "%.2f%%", testAccuracy * 100))")
            } else {
                print("Epoch \(epoch): Train Loss = \(String(format: "%.4f", avgEpochLoss))")
            }
        }
    }
}
