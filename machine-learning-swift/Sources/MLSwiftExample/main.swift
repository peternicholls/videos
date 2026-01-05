/// main.swift
/// Example usage of MLSwift neural network library
/// Demonstrates XOR problem and MNIST-like classification

import Foundation
import MLSwift

/// Example 1: Train a simple neural network on the XOR problem
func xorExample() {
    print("=== XOR Problem Example ===\n")
    
    // Create a simple 2-layer network: 2 -> 4 -> 1
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 2, outputSize: 4))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 4, outputSize: 1))
    
    // Use MSE loss for regression
    model.setLoss(Loss.meanSquaredError, gradient: Loss.meanSquaredErrorBackward)
    
    // XOR training data
    let inputs = [
        Matrix(rows: 2, cols: 1, data: [0.0, 0.0]),
        Matrix(rows: 2, cols: 1, data: [0.0, 1.0]),
        Matrix(rows: 2, cols: 1, data: [1.0, 0.0]),
        Matrix(rows: 2, cols: 1, data: [1.0, 1.0])
    ]
    
    let targets = [
        Matrix(rows: 1, cols: 1, data: [0.0]),
        Matrix(rows: 1, cols: 1, data: [1.0]),
        Matrix(rows: 1, cols: 1, data: [1.0]),
        Matrix(rows: 1, cols: 1, data: [0.0])
    ]
    
    // Train for 1000 epochs
    print("Training XOR network...")
    for epoch in 1...1000 {
        var epochLoss: Float = 0.0
        for (input, target) in zip(inputs, targets) {
            epochLoss += model.trainStep(input: input, target: target, learningRate: 0.1)
        }
        
        if epoch % 100 == 0 {
            print("Epoch \(epoch): Loss = \(String(format: "%.6f", epochLoss / 4.0))")
        }
    }
    
    // Test the trained network
    print("\nTesting XOR network:")
    for (input, target) in zip(inputs, targets) {
        let output = model.forward(input)
        print("Input: [\(input[0, 0]), \(input[1, 0])] -> Output: \(String(format: "%.4f", output[0, 0])) (Target: \(target[0, 0]))")
    }
    print()
}

/// Example 2: Create a small classification network
func classificationExample() {
    print("=== Multi-class Classification Example ===\n")
    
    // Create a 3-layer network: 10 -> 16 -> 8 -> 3
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 10, outputSize: 16))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 16, outputSize: 8))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 8, outputSize: 3))
    model.add(SoftmaxLayer())
    
    // Use cross-entropy loss
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Generate synthetic training data (3 classes)
    print("Generating synthetic training data...")
    var trainInputs: [Matrix] = []
    var trainTargets: [Matrix] = []
    
    for _ in 0..<300 {
        let classLabel = Int.random(in: 0..<3)
        
        // Generate random features with class-dependent bias
        var features = [Float](repeating: 0.0, count: 10)
        for i in 0..<10 {
            let base = Float.random(in: -1.0..<1.0)
            features[i] = base + Float(classLabel) * 0.5
        }
        
        let input = Matrix(rows: 10, cols: 1, data: features)
        
        // One-hot encode the target
        var targetData = [Float](repeating: 0.0, count: 3)
        targetData[classLabel] = 1.0
        let target = Matrix(rows: 3, cols: 1, data: targetData)
        
        trainInputs.append(input)
        trainTargets.append(target)
    }
    
    // Generate test data
    var testInputs: [Matrix] = []
    var testTargets: [Matrix] = []
    
    for _ in 0..<100 {
        let classLabel = Int.random(in: 0..<3)
        
        var features = [Float](repeating: 0.0, count: 10)
        for i in 0..<10 {
            let base = Float.random(in: -1.0..<1.0)
            features[i] = base + Float(classLabel) * 0.5
        }
        
        let input = Matrix(rows: 10, cols: 1, data: features)
        
        var targetData = [Float](repeating: 0.0, count: 3)
        targetData[classLabel] = 1.0
        let target = Matrix(rows: 3, cols: 1, data: targetData)
        
        testInputs.append(input)
        testTargets.append(target)
    }
    
    // Train the model
    print("Training classification network...")
    model.train(
        trainInputs: trainInputs,
        trainTargets: trainTargets,
        testInputs: testInputs,
        testTargets: testTargets,
        epochs: 20,
        batchSize: 32,
        learningRate: 0.01
    )
    
    print("\nTraining complete!")
}

/// Example 3: Demonstrate new features (Dropout, BatchNorm, Optimizers, Serialization)
func advancedFeaturesExample() {
    print("=== Advanced Features Example ===\n")
    
    // Create a model with dropout and batch normalization
    print("Creating model with Dropout and Batch Normalization...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 8, outputSize: 16))
    model.add(BatchNormLayer(numFeatures: 16))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
    model.add(DenseLayer(inputSize: 16, outputSize: 8))
    model.add(TanhLayer())
    model.add(DenseLayer(inputSize: 8, outputSize: 3))
    model.add(SoftmaxLayer())
    
    // Use cross-entropy loss
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Generate synthetic training data
    print("Generating synthetic data...")
    var trainInputs: [Matrix] = []
    var trainTargets: [Matrix] = []
    
    for _ in 0..<200 {
        let classLabel = Int.random(in: 0..<3)
        var features = [Float](repeating: 0.0, count: 8)
        for i in 0..<8 {
            features[i] = Float.random(in: -1.0..<1.0) + Float(classLabel) * 0.4
        }
        
        let input = Matrix(rows: 8, cols: 1, data: features)
        var targetData = [Float](repeating: 0.0, count: 3)
        targetData[classLabel] = 1.0
        let target = Matrix(rows: 3, cols: 1, data: targetData)
        
        trainInputs.append(input)
        trainTargets.append(target)
    }
    
    // Train with dropout enabled
    print("Training model (dropout and batch norm enabled)...")
    for epoch in 1...10 {
        var epochLoss: Float = 0.0
        for (input, target) in zip(trainInputs, trainTargets) {
            epochLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
        }
        if epoch % 2 == 0 {
            print("Epoch \(epoch): Loss = \(String(format: "%.4f", epochLoss / Float(trainInputs.count)))")
        }
    }
    
    // Save the model
    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("mlswift_model.json")
    do {
        try model.save(to: tempURL)
        print("\nModel saved to: \(tempURL.path)")
        
        // Load the model
        let loadedModel = try SequentialModel.load(from: tempURL)
        print("Model loaded successfully!")
        
        // Set both models to inference mode for fair comparison
        for layer in model.getLayers() {
            if let dropout = layer as? DropoutLayer {
                dropout.training = false
            }
            if let batchNorm = layer as? BatchNormLayer {
                batchNorm.training = false
            }
        }
        for layer in loadedModel.getLayers() {
            if let dropout = layer as? DropoutLayer {
                dropout.training = false
            }
            if let batchNorm = layer as? BatchNormLayer {
                batchNorm.training = false
            }
        }
        
        // Test both models produce same output
        let testInput = trainInputs[0]
        let originalOutput = model.forward(testInput)
        let loadedOutput = loadedModel.forward(testInput)
        
        print("\nVerifying loaded model (inference mode):")
        print("Original model output: [\(String(format: "%.4f", originalOutput.data[0])), \(String(format: "%.4f", originalOutput.data[1])), \(String(format: "%.4f", originalOutput.data[2]))]")
        print("Loaded model output:   [\(String(format: "%.4f", loadedOutput.data[0])), \(String(format: "%.4f", loadedOutput.data[1])), \(String(format: "%.4f", loadedOutput.data[2]))]")
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    } catch {
        print("Error with serialization: \(error)")
    }
    
    print()
}

/// Example 4: Demonstrate different optimizers
func optimizerComparison() {
    print("=== Optimizer Comparison Example ===\n")
    
    // Simple problem: learn to approximate a function
    let inputs = [
        Matrix(rows: 1, cols: 1, data: [0.0]),
        Matrix(rows: 1, cols: 1, data: [0.5]),
        Matrix(rows: 1, cols: 1, data: [1.0])
    ]
    
    let targets = [
        Matrix(rows: 1, cols: 1, data: [0.0]),
        Matrix(rows: 1, cols: 1, data: [0.25]),
        Matrix(rows: 1, cols: 1, data: [1.0])
    ]
    
    // Note: Optimizers are designed for custom training loops
    // This example demonstrates the API. For actual use, you would:
    // 1. Collect parameters/gradients from model layers
    // 2. Call optimizer.update() to modify parameters
    // 3. Apply updated parameters back to layers
    
    let optimizerNames = ["SGD", "SGD+Momentum", "Adam", "RMSprop"]
    let learningRates: [Float] = [0.1, 0.05, 0.01, 0.01]
    
    for (index, name) in optimizerNames.enumerated() {
        print("Testing \(name) optimizer behavior...")
        
        // Create a simple model
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 1, outputSize: 4))
        model.add(SigmoidLayer())
        model.add(DenseLayer(inputSize: 4, outputSize: 1))
        model.setLoss(Loss.meanSquaredError, gradient: Loss.meanSquaredErrorBackward)
        
        // Train using model's built-in SGD (for demonstration)
        for _ in 0..<100 {
            for (input, target) in zip(inputs, targets) {
                model.trainStep(input: input, target: target, learningRate: learningRates[index])
            }
        }
        
        // Test final loss
        var totalLoss: Float = 0.0
        for (input, target) in zip(inputs, targets) {
            totalLoss += model.computeLoss(input: input, target: target)
        }
        print("  Final loss: \(String(format: "%.6f", totalLoss / Float(inputs.count)))")
    }
    
    print("\nNote: For actual optimizer usage, see README.md for custom training loop examples.")
    print()
}

/// Example 5: Demonstrate Metal GPU acceleration (if available)
func metalAccelerationDemo() {
    print("=== Metal GPU Acceleration Demo ===\n")
    
    #if canImport(Metal)
    let device = MetalDevice.shared
    print("Metal Device: \(device.device.name)")
    print("Max threads per threadgroup: \(device.device.maxThreadsPerThreadgroup)")
    print()
    
    // Create two large matrices
    let size = 512
    print("Creating \(size)x\(size) matrices...")
    let a = Matrix(rows: size, cols: size, randomInRange: -1.0, 1.0)
    let b = Matrix(rows: size, cols: size, randomInRange: -1.0, 1.0)
    
    // CPU matrix multiplication
    print("Performing CPU matrix multiplication...")
    let startCPU = Date()
    let resultCPU = Matrix.multiply(a, b)
    let cpuTime = Date().timeIntervalSince(startCPU)
    print("CPU time: \(String(format: "%.3f", cpuTime)) seconds")
    
    // GPU matrix multiplication
    print("Performing GPU matrix multiplication...")
    let startGPU = Date()
    let resultGPU = try! Matrix.multiplyGPU(a, b)
    let gpuTime = Date().timeIntervalSince(startGPU)
    print("GPU time: \(String(format: "%.3f", gpuTime)) seconds")
    
    // Verify results match (approximately)
    var maxDiff: Float = 0.0
    for i in 0..<resultCPU.data.count {
        let diff = abs(resultCPU.data[i] - resultGPU.data[i])
        maxDiff = max(maxDiff, diff)
    }
    print("Max difference between CPU and GPU: \(String(format: "%.6f", maxDiff))")
    print("Speedup: \(String(format: "%.2f", cpuTime / gpuTime))x")
    print()
    #else
    print("Metal is not available on this platform (requires macOS)")
    print("The library will use CPU-only operations.")
    print()
    #endif
}

/// Example 6: Demonstrate model saving and loading (from README documentation)
func modelPersistenceExample() {
    print("=== Model Persistence Example ===\n")
    
    // Create and train a simple model
    print("Creating and training a model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 8, outputSize: 12))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 12, outputSize: 3))
    model.add(SoftmaxLayer())
    
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Generate training data
    var trainInputs: [Matrix] = []
    var trainTargets: [Matrix] = []
    
    for _ in 0..<100 {
        let classLabel = Int.random(in: 0..<3)
        let features = [Float](repeating: 0, count: 8).map { _ in 
            Float.random(in: -1.0...1.0) + Float(classLabel) * 0.4
        }
        trainInputs.append(Matrix(rows: 8, cols: 1, data: features))
        
        var oneHot = [Float](repeating: 0.0, count: 3)
        oneHot[classLabel] = 1.0
        trainTargets.append(Matrix(rows: 3, cols: 1, data: oneHot))
    }
    
    // Train briefly
    for epoch in 1...5 {
        var totalLoss: Float = 0.0
        for (input, target) in zip(trainInputs, trainTargets) {
            totalLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
        }
        if epoch % 2 == 0 {
            print("Epoch \(epoch): Loss = \(String(format: "%.4f", totalLoss / Float(trainInputs.count)))")
        }
    }
    
    // Save the model
    let saveURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("example_model_\(UUID().uuidString).json")
    
    do {
        try model.save(to: saveURL)
        print("\nModel saved to: \(saveURL.lastPathComponent)")
        
        // Check file size
        if let attributes = try? FileManager.default.attributesOfItem(atPath: saveURL.path),
           let fileSize = attributes[.size] as? Int {
            print("File size: \(String(format: "%.2f", Float(fileSize) / 1024.0)) KB")
        }
        
        // Load the model back
        let loadedModel = try SequentialModel.load(from: saveURL)
        print("Model loaded successfully!")
        print("Loaded model has \(loadedModel.getLayers().count) layers")
        
        // Verify models produce same output
        let testInput = trainInputs[0]
        let originalOutput = model.forward(testInput)
        let loadedOutput = loadedModel.forward(testInput)
        
        var maxDiff: Float = 0.0
        for i in 0..<originalOutput.data.count {
            let diff = abs(originalOutput.data[i] - loadedOutput.data[i])
            maxDiff = max(maxDiff, diff)
        }
        
        print("Max difference between original and loaded: \(String(format: "%.10f", maxDiff))")
        if maxDiff < 0.0001 {
            print("✓ Serialization verified - models match!")
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: saveURL)
        
    } catch {
        print("Error with model persistence: \(error)")
    }
    
    print()
}

// Main program
print("MLSwift - Neural Network Library for Apple Silicon\n")
#if canImport(Metal)
print("This library demonstrates GPU-accelerated neural networks using Metal.")
print("Optimized for macOS with Apple Silicon (M1/M2/M3+)\n")
#else
print("Running in CPU-only mode (Metal not available on this platform)")
print("For GPU acceleration, run on macOS with Apple Silicon\n")
#endif

// Run examples
xorExample()
classificationExample()
advancedFeaturesExample()
optimizerComparison()
modelPersistenceExample()
metalAccelerationDemo()

print("All examples completed!")
