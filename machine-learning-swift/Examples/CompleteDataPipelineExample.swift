/// CompleteDataPipelineExample.swift
/// Demonstrates the complete data pipeline using all new utilities
/// Shows how to load, preprocess, train, save, and use models

import Foundation
import MLSwift

/// Complete example demonstrating data loading, preprocessing, training, and model persistence
func completeDataPipelineExample() {
    print("=== Complete Data Pipeline Example ===")
    print("Demonstrates: Data loading → Preprocessing → Training → Saving → Loading → Inference\n")
    
    // STEP 1: Generate synthetic dataset (in production, you'd load from files)
    print("Step 1: Generating synthetic dataset...")
    var rawInputs: [Matrix] = []
    var rawLabels: [Int] = []
    
    for i in 0..<500 {
        let classLabel = i % 3
        let features = [Float](repeating: 0, count: 20).map { _ in
            Float.random(in: 0...100) + Float(classLabel) * 20.0
        }
        rawInputs.append(Matrix(rows: 20, cols: 1, data: features))
        rawLabels.append(classLabel)
    }
    print("✓ Generated 500 samples with 20 features each, 3 classes")
    
    // STEP 2: Normalize the data
    print("\nStep 2: Normalizing data...")
    let normalizedInputs = rawInputs.map { minMaxNormalize($0) }
    print("✓ Normalized features to [0, 1] range")
    
    // STEP 3: One-hot encode labels
    print("\nStep 3: Encoding labels...")
    let encodedLabels = oneHotEncode(labels: rawLabels, numClasses: 3)
    print("✓ Encoded labels to one-hot format")
    
    // STEP 4: Split into train/validation
    print("\nStep 4: Splitting dataset...")
    let (trainInputs, valInputs) = trainValidationSplit(data: normalizedInputs, splitRatio: 0.8)
    let (trainLabels, valLabels) = trainValidationSplit(data: encodedLabels, splitRatio: 0.8)
    print("✓ Training: \(trainInputs.count) samples")
    print("✓ Validation: \(valInputs.count) samples")
    
    // STEP 5: Shuffle training data
    print("\nStep 5: Shuffling training data...")
    let (shuffledInputs, shuffledLabels) = shuffleDataset(inputs: trainInputs, targets: trainLabels)
    print("✓ Training data shuffled")
    
    // STEP 6: Create model
    print("\nStep 6: Creating neural network model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 20, outputSize: 32))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 32, outputSize: 16))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 16, outputSize: 3))
    model.add(SoftmaxLayer())
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    print("✓ Model created: 20 → 32 → 16 → 3")
    
    // STEP 7: Train the model
    print("\nStep 7: Training model...")
    let batches = createBatches(inputs: shuffledInputs, targets: shuffledLabels, batchSize: 32)
    
    for epoch in 1...10 {
        var epochLoss: Float = 0.0
        var batchCount = 0
        
        for (batchInputs, batchTargets) in batches {
            for (input, target) in zip(batchInputs, batchTargets) {
                epochLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
                batchCount += 1
            }
        }
        
        if epoch % 2 == 0 {
            let avgLoss = epochLoss / Float(batchCount)
            print("  Epoch \(epoch)/10: Loss = \(String(format: "%.4f", avgLoss))")
        }
    }
    print("✓ Training complete")
    
    // STEP 8: Evaluate on validation set
    print("\nStep 8: Evaluating on validation set...")
    var correct = 0
    for (input, target) in zip(valInputs, valLabels) {
        let prediction = model.forward(input)
        let predictedClass = prediction.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        let actualClass = target.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        if predictedClass == actualClass {
            correct += 1
        }
    }
    let accuracy = Float(correct) / Float(valInputs.count) * 100.0
    print("✓ Validation Accuracy: \(String(format: "%.2f", accuracy))%")
    
    // STEP 9: Save the trained model
    print("\nStep 9: Saving trained model...")
    let modelURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("complete_pipeline_model.json")
    
    do {
        try model.save(to: modelURL)
        let fileSize = try FileManager.default.attributesOfItem(atPath: modelURL.path)[.size] as? Int ?? 0
        print("✓ Model saved: \(modelURL.lastPathComponent)")
        print("  File size: \(String(format: "%.2f", Float(fileSize) / 1024.0)) KB")
    } catch {
        print("✗ Error saving model: \(error)")
        return
    }
    
    // STEP 10: Load the model for inference
    print("\nStep 10: Loading model for inference...")
    do {
        let loadedModel = try SequentialModel.load(from: modelURL)
        print("✓ Model loaded successfully")
        
        // STEP 11: Use loaded model for prediction
        print("\nStep 11: Making predictions with loaded model...")
        let testSample = valInputs[0]
        let actualLabel = oneHotDecode([valLabels[0]])[0]
        let prediction = loadedModel.forward(testSample)
        let predictedClass = prediction.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        
        print("Test sample prediction:")
        print("  Actual class: \(actualLabel)")
        print("  Predicted class: \(predictedClass)")
        print("  Probabilities:")
        for i in 0..<prediction.data.count {
            print("    Class \(i): \(String(format: "%.4f", prediction.data[i])) (\(String(format: "%.1f", prediction.data[i] * 100))%)")
        }
        
        if predictedClass == actualLabel {
            print("  ✓ Correct prediction!")
        } else {
            print("  ✗ Incorrect prediction")
        }
        
    } catch {
        print("✗ Error loading model: \(error)")
    }
    
    // Clean up
    try? FileManager.default.removeItem(at: modelURL)
    
    print("\n=== Pipeline Complete! ===\n")
}

/// Example showing data file operations (compatible with Python)
func dataFileOperationsExample() {
    print("=== Data File Operations Example ===")
    print("Demonstrates saving/loading data in Python-compatible format\n")
    
    // Create sample data
    let images = (0..<10).map { i in
        Matrix(rows: 28, cols: 28, data: [Float](repeating: Float(i), count: 784))
    }
    let labels = Array(0..<10)
    
    print("Step 1: Saving data to binary files...")
    let tempDir = FileManager.default.temporaryDirectory
    let imagesURL = tempDir.appendingPathComponent("test_images.mat")
    let labelsURL = tempDir.appendingPathComponent("test_labels.mat")
    
    do {
        // Flatten all images into one big matrix
        let allImagesData = images.flatMap { $0.data }
        let imagesMatrix = Matrix(rows: images.count, cols: 784, data: allImagesData)
        try saveBinaryMatrix(imagesMatrix, to: imagesURL)
        
        let labelsMatrix = Matrix(rows: labels.count, cols: 1, data: labels.map { Float($0) })
        try saveBinaryMatrix(labelsMatrix, to: labelsURL)
        
        print("✓ Images saved: \(imagesURL.lastPathComponent)")
        print("✓ Labels saved: \(labelsURL.lastPathComponent)")
        
        // Load them back
        print("\nStep 2: Loading data from binary files...")
        let loadedImages = try loadBinaryMatrix(from: imagesURL, rows: 10, cols: 784)
        let loadedLabels = try loadBinaryMatrix(from: labelsURL, rows: 10, cols: 1)
        
        print("✓ Images loaded: \(loadedImages.rows) samples, \(loadedImages.cols) features")
        print("✓ Labels loaded: \(loadedLabels.rows) samples")
        
        // Verify
        print("\nStep 3: Verifying data integrity...")
        var matches = true
        for i in 0..<min(10, loadedImages.data.count) {
            if abs(loadedImages.data[i] - allImagesData[i]) > 0.001 {
                matches = false
                break
            }
        }
        
        if matches {
            print("✓ Data integrity verified - files match!")
        } else {
            print("✗ Data mismatch detected")
        }
        
    } catch {
        print("✗ Error: \(error)")
    }
    
    // Clean up
    try? FileManager.default.removeItem(at: imagesURL)
    try? FileManager.default.removeItem(at: labelsURL)
    
    print("\n=== File Operations Complete! ===\n")
}

#if canImport(Accelerate)
import Accelerate

/// Example demonstrating Accelerate framework integration (macOS/iOS only)
func accelerateIntegrationExample() {
    print("=== Accelerate Framework Integration Example ===")
    print("High-performance operations using vDSP and BLAS\n")
    
    // Example 1: Normalization with Accelerate
    print("Example 1: Fast normalization with vDSP")
    let data = Matrix(rows: 100, cols: 100, randomInRange: 0, 100)
    let (normalized, mean, stdDev) = normalizeWithAccelerate(data)
    print("  Original mean: \(String(format: "%.2f", mean))")
    print("  Original std dev: \(String(format: "%.2f", stdDev))")
    print("  Normalized mean: \(String(format: "%.6f", meanWithAccelerate(normalized)))")
    print("✓ Normalization complete\n")
    
    // Example 2: Matrix multiplication with BLAS
    print("Example 2: Matrix multiplication with BLAS")
    let a = Matrix(rows: 100, cols: 50, randomInRange: -1, 1)
    let b = Matrix(rows: 50, cols: 75, randomInRange: -1, 1)
    
    let startTime = Date()
    if let result = matrixMultiplyWithBLAS(a, b) {
        let elapsed = Date().timeIntervalSince(startTime)
        print("  Multiplied \(a.rows)×\(a.cols) by \(b.rows)×\(b.cols)")
        print("  Result: \(result.rows)×\(result.cols)")
        print("  Time: \(String(format: "%.4f", elapsed)) seconds")
        print("✓ BLAS multiplication complete\n")
    }
    
    // Example 3: Statistical operations
    print("Example 3: Fast statistics with vDSP")
    let values = Matrix(rows: 1000, cols: 1, randomInRange: 0, 100)
    print("  Mean: \(String(format: "%.2f", meanWithAccelerate(values)))")
    print("  Sum: \(String(format: "%.2f", sumWithAccelerate(values)))")
    print("  Max: \(String(format: "%.2f", maxWithAccelerate(values)))")
    print("  Min: \(String(format: "%.2f", minWithAccelerate(values)))")
    print("✓ Statistics complete\n")
    
    print("=== Accelerate Examples Complete! ===\n")
}
#endif
