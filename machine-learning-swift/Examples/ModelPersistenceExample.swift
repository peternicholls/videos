/// ModelPersistenceExample.swift
/// Demonstrates model saving, loading, and usage
/// This example validates the serialization functionality

import Foundation

#if canImport(MLSwift)
import MLSwift

/// Example demonstrating the complete model lifecycle
func demonstrateModelPersistence() {
    print("=== Model Persistence Demonstration ===\n")
    
    // STEP 1: Create and train a model
    print("Step 1: Creating and training model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 10, outputSize: 16))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
    model.add(DenseLayer(inputSize: 16, outputSize: 8))
    model.add(BatchNormLayer(numFeatures: 8))
    model.add(TanhLayer())
    model.add(DenseLayer(inputSize: 8, outputSize: 3))
    model.add(SoftmaxLayer())
    
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Generate synthetic training data
    var trainInputs: [Matrix] = []
    var trainTargets: [Matrix] = []
    
    for _ in 0..<200 {
        let classLabel = Int.random(in: 0..<3)
        let features = [Float](repeating: 0, count: 10).map { _ in 
            Float.random(in: -1.0...1.0) + Float(classLabel) * 0.5
        }
        trainInputs.append(Matrix(rows: 10, cols: 1, data: features))
        
        var oneHot = [Float](repeating: 0.0, count: 3)
        oneHot[classLabel] = 1.0
        trainTargets.append(Matrix(rows: 3, cols: 1, data: oneHot))
    }
    
    // Train the model
    print("Training for 5 epochs...")
    for epoch in 1...5 {
        var totalLoss: Float = 0.0
        for (input, target) in zip(trainInputs, trainTargets) {
            totalLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
        }
        print("  Epoch \(epoch): Loss = \(String(format: "%.4f", totalLoss / Float(trainInputs.count)))")
    }
    
    // STEP 2: Save the trained model
    print("\nStep 2: Saving trained model...")
    let saveURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("test_model_\(UUID().uuidString).json")
    
    do {
        try model.save(to: saveURL)
        print("✓ Model saved to: \(saveURL.path)")
        
        let attributes = try FileManager.default.attributesOfItem(atPath: saveURL.path)
        if let fileSize = attributes[.size] as? Int {
            print("  File size: \(fileSize) bytes (\(String(format: "%.2f", Float(fileSize) / 1024.0)) KB)")
        }
    } catch {
        print("✗ Error saving model: \(error)")
        return
    }
    
    // STEP 3: Load the model from disk
    print("\nStep 3: Loading model from disk...")
    let loadedModel: SequentialModel
    do {
        loadedModel = try SequentialModel.load(from: saveURL)
        print("✓ Model loaded successfully!")
        print("  Number of layers: \(loadedModel.getLayers().count)")
        
        // Print layer information
        for (i, layer) in loadedModel.getLayers().enumerated() {
            let layerType = String(describing: type(of: layer))
                .replacingOccurrences(of: "MLSwift.", with: "")
            print("  Layer \(i + 1): \(layerType)")
        }
    } catch {
        print("✗ Error loading model: \(error)")
        try? FileManager.default.removeItem(at: saveURL)
        return
    }
    
    // STEP 4: Set models to inference mode
    print("\nStep 4: Configuring models for inference...")
    
    func setInferenceMode(_ model: SequentialModel) {
        for layer in model.getLayers() {
            if let dropout = layer as? DropoutLayer {
                dropout.training = false
            }
            if let batchNorm = layer as? BatchNormLayer {
                batchNorm.training = false
            }
        }
    }
    
    setInferenceMode(model)
    setInferenceMode(loadedModel)
    print("✓ Both models set to inference mode")
    
    // STEP 5: Compare outputs
    print("\nStep 5: Verifying model integrity...")
    var allTestsPassed = true
    var maxDiff: Float = 0.0
    
    // Test with multiple inputs
    for i in 0..<5 {
        let testInput = trainInputs[i]
        let originalOutput = model.forward(testInput)
        let loadedOutput = loadedModel.forward(testInput)
        
        var testMaxDiff: Float = 0.0
        for j in 0..<originalOutput.data.count {
            let diff = abs(originalOutput.data[j] - loadedOutput.data[j])
            testMaxDiff = max(testMaxDiff, diff)
            maxDiff = max(maxDiff, diff)
        }
        
        if testMaxDiff > 0.0001 {
            allTestsPassed = false
            print("  Test \(i + 1): ⚠ Difference = \(String(format: "%.10f", testMaxDiff))")
        } else {
            print("  Test \(i + 1): ✓ Outputs match (diff = \(String(format: "%.10f", testMaxDiff)))")
        }
    }
    
    print("\nOverall max difference: \(String(format: "%.10f", maxDiff))")
    if allTestsPassed {
        print("✓ All tests passed! Model serialization is working correctly.")
    } else {
        print("⚠ Some tests showed differences (may be acceptable for floating-point)")
    }
    
    // STEP 6: Demonstrate inference usage
    print("\nStep 6: Making predictions with loaded model...")
    let newInput = Matrix(rows: 10, cols: 1, data: [Float](repeating: 0, count: 10).map { _ in Float.random(in: -1...1) })
    let prediction = loadedModel.forward(newInput)
    
    print("Input features: [\(newInput.data.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", ")), ...]")
    print("Prediction probabilities:")
    for i in 0..<prediction.data.count {
        print("  Class \(i): \(String(format: "%.4f", prediction.data[i])) (\(String(format: "%.1f", prediction.data[i] * 100))%)")
    }
    
    let predictedClass = prediction.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
    print("Predicted class: \(predictedClass)")
    
    // STEP 7: Demonstrate continuing training
    print("\nStep 7: Continuing training with loaded model...")
    
    // Re-enable training mode
    for layer in loadedModel.getLayers() {
        if let dropout = layer as? DropoutLayer {
            dropout.training = true
        }
        if let batchNorm = layer as? BatchNormLayer {
            batchNorm.training = true
        }
    }
    
    // Set loss function (not persisted)
    loadedModel.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Train for a few more epochs
    print("Fine-tuning for 3 more epochs...")
    for epoch in 1...3 {
        var totalLoss: Float = 0.0
        for (input, target) in zip(trainInputs.prefix(50), trainTargets.prefix(50)) {
            totalLoss += loadedModel.trainStep(input: input, target: target, learningRate: 0.005)
        }
        print("  Epoch \(epoch): Loss = \(String(format: "%.4f", totalLoss / 50.0))")
    }
    
    // Save fine-tuned model
    let fineTunedURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("test_model_finetuned_\(UUID().uuidString).json")
    
    do {
        try loadedModel.save(to: fineTunedURL)
        print("✓ Fine-tuned model saved to: \(fineTunedURL.path)")
    } catch {
        print("✗ Error saving fine-tuned model: \(error)")
    }
    
    // Clean up
    print("\nStep 8: Cleaning up temporary files...")
    try? FileManager.default.removeItem(at: saveURL)
    try? FileManager.default.removeItem(at: fineTunedURL)
    print("✓ Temporary files removed")
    
    print("\n=== Model Persistence Demonstration Complete! ===\n")
}

/// Quick validation test
func validateSerializationBasics() {
    print("=== Quick Serialization Validation ===\n")
    
    // Test 1: Simple model without dropout/batchnorm
    print("Test 1: Simple model (Dense + ReLU + Softmax)")
    let simpleModel = SequentialModel()
    simpleModel.add(DenseLayer(inputSize: 5, outputSize: 3))
    simpleModel.add(ReLULayer())
    simpleModel.add(DenseLayer(inputSize: 3, outputSize: 2))
    simpleModel.add(SoftmaxLayer())
    
    let simpleURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("simple_test.json")
    
    do {
        try simpleModel.save(to: simpleURL)
        let loaded = try SequentialModel.load(from: simpleURL)
        print("✓ Simple model save/load successful")
        try? FileManager.default.removeItem(at: simpleURL)
    } catch {
        print("✗ Simple model test failed: \(error)")
    }
    
    // Test 2: Complex model with all layer types
    print("\nTest 2: Complex model (all supported layer types)")
    let complexModel = SequentialModel()
    complexModel.add(DenseLayer(inputSize: 10, outputSize: 8))
    complexModel.add(BatchNormLayer(numFeatures: 8))
    complexModel.add(ReLULayer())
    complexModel.add(DropoutLayer(dropoutRate: 0.5))
    complexModel.add(DenseLayer(inputSize: 8, outputSize: 4))
    complexModel.add(SigmoidLayer())
    complexModel.add(DenseLayer(inputSize: 4, outputSize: 2))
    complexModel.add(TanhLayer())
    
    let complexURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("complex_test.json")
    
    do {
        try complexModel.save(to: complexURL)
        let loaded = try SequentialModel.load(from: complexURL)
        print("✓ Complex model save/load successful")
        print("  Loaded \(loaded.getLayers().count) layers")
        try? FileManager.default.removeItem(at: complexURL)
    } catch {
        print("✗ Complex model test failed: \(error)")
    }
    
    print("\n=== Validation Complete ===\n")
}

#endif
