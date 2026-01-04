/// MLSwiftTests.swift
/// Unit tests for MLSwift neural network library

import XCTest
@testable import MLSwift

final class MatrixTests: XCTestCase {
    
    func testMatrixCreation() {
        let mat = Matrix(rows: 3, cols: 2)
        XCTAssertEqual(mat.rows, 3)
        XCTAssertEqual(mat.cols, 2)
        XCTAssertEqual(mat.count, 6)
        XCTAssertEqual(mat.data.count, 6)
        
        // Should be zero-initialized
        for value in mat.data {
            XCTAssertEqual(value, 0.0)
        }
    }
    
    func testMatrixWithValue() {
        let mat = Matrix(rows: 2, cols: 3, value: 5.0)
        XCTAssertEqual(mat.rows, 2)
        XCTAssertEqual(mat.cols, 3)
        
        for value in mat.data {
            XCTAssertEqual(value, 5.0)
        }
    }
    
    func testMatrixSubscript() {
        var mat = Matrix(rows: 2, cols: 2)
        mat[0, 0] = 1.0
        mat[0, 1] = 2.0
        mat[1, 0] = 3.0
        mat[1, 1] = 4.0
        
        XCTAssertEqual(mat[0, 0], 1.0)
        XCTAssertEqual(mat[0, 1], 2.0)
        XCTAssertEqual(mat[1, 0], 3.0)
        XCTAssertEqual(mat[1, 1], 4.0)
    }
    
    func testMatrixSum() {
        let mat = Matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
        XCTAssertEqual(mat.sum(), 10.0)
    }
    
    func testMatrixArgmax() {
        let mat = Matrix(rows: 1, cols: 4, data: [1.0, 5.0, 3.0, 2.0])
        XCTAssertEqual(mat.argmax(), 1)
    }
    
    func testMatrixAddition() {
        let a = Matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
        let b = Matrix(rows: 2, cols: 2, data: [5.0, 6.0, 7.0, 8.0])
        let c = Matrix.add(a, b)
        
        XCTAssertEqual(c.data, [6.0, 8.0, 10.0, 12.0])
    }
    
    func testMatrixSubtraction() {
        let a = Matrix(rows: 2, cols: 2, data: [5.0, 6.0, 7.0, 8.0])
        let b = Matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
        let c = Matrix.subtract(a, b)
        
        XCTAssertEqual(c.data, [4.0, 4.0, 4.0, 4.0])
    }
    
    func testMatrixMultiplication() {
        // Test 2x3 * 3x2 = 2x2
        let a = Matrix(rows: 2, cols: 3, data: [1.0, 2.0, 3.0,
                                                  4.0, 5.0, 6.0])
        let b = Matrix(rows: 3, cols: 2, data: [7.0, 8.0,
                                                  9.0, 10.0,
                                                  11.0, 12.0])
        let c = Matrix.multiply(a, b)
        
        // Expected: [[58, 64], [139, 154]]
        XCTAssertEqual(c.rows, 2)
        XCTAssertEqual(c.cols, 2)
        XCTAssertEqual(c[0, 0], 58.0, accuracy: 0.001)
        XCTAssertEqual(c[0, 1], 64.0, accuracy: 0.001)
        XCTAssertEqual(c[1, 0], 139.0, accuracy: 0.001)
        XCTAssertEqual(c[1, 1], 154.0, accuracy: 0.001)
    }
    
    func testMatrixMultiplicationTranspose() {
        let a = Matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
        let b = Matrix(rows: 2, cols: 2, data: [5.0, 6.0, 7.0, 8.0])
        
        // A^T * B
        let c1 = Matrix.multiply(a, b, transposeA: true)
        XCTAssertEqual(c1.rows, 2)
        XCTAssertEqual(c1.cols, 2)
        
        // A * B^T
        let c2 = Matrix.multiply(a, b, transposeB: true)
        XCTAssertEqual(c2.rows, 2)
        XCTAssertEqual(c2.cols, 2)
    }
}

final class ActivationTests: XCTestCase {
    
    func testReLU() {
        let input = Matrix(rows: 1, cols: 4, data: [-2.0, -1.0, 0.0, 1.0])
        let output = Activations.relu(input)
        
        XCTAssertEqual(output.data, [0.0, 0.0, 0.0, 1.0])
    }
    
    func testReLUBackward() {
        let input = Matrix(rows: 1, cols: 4, data: [-2.0, -1.0, 0.5, 1.0])
        let gradOutput = Matrix(rows: 1, cols: 4, value: 1.0)
        let gradInput = Activations.reluBackward(input: input, gradOutput: gradOutput)
        
        XCTAssertEqual(gradInput.data, [0.0, 0.0, 1.0, 1.0])
    }
    
    func testSoftmax() {
        let input = Matrix(rows: 3, cols: 1, data: [1.0, 2.0, 3.0])
        let output = Activations.softmax(input)
        
        // Sum should be 1
        let sum = output.sum()
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
        
        // All values should be positive
        for value in output.data {
            XCTAssertTrue(value > 0.0)
        }
        
        // Largest input should give largest output
        XCTAssertTrue(output.data[2] > output.data[1])
        XCTAssertTrue(output.data[1] > output.data[0])
    }
    
    func testSigmoid() {
        let input = Matrix(rows: 1, cols: 3, data: [-1.0, 0.0, 1.0])
        let output = Activations.sigmoid(input)
        
        // All values should be in (0, 1)
        for value in output.data {
            XCTAssertTrue(value > 0.0 && value < 1.0)
        }
        
        // sigmoid(0) = 0.5
        XCTAssertEqual(output.data[1], 0.5, accuracy: 0.001)
        
        // sigmoid(-x) = 1 - sigmoid(x)
        XCTAssertEqual(output.data[0], 1.0 - output.data[2], accuracy: 0.001)
    }
}

final class LossTests: XCTestCase {
    
    func testCrossEntropy() {
        let predicted = Matrix(rows: 3, cols: 1, data: [0.1, 0.2, 0.7])
        let target = Matrix(rows: 3, cols: 1, data: [0.0, 0.0, 1.0])
        
        let loss = Loss.crossEntropy(predicted: predicted, target: target)
        
        // Loss should be positive
        XCTAssertTrue(loss > 0.0)
        
        // Expected: -1.0 * log(0.7) ≈ 0.357
        XCTAssertEqual(loss, -log(0.7), accuracy: 0.001)
    }
    
    func testMeanSquaredError() {
        let predicted = Matrix(rows: 2, cols: 1, data: [1.0, 2.0])
        let target = Matrix(rows: 2, cols: 1, data: [1.5, 2.5])
        
        let loss = Loss.meanSquaredError(predicted: predicted, target: target)
        
        // Expected: ((1-1.5)^2 + (2-2.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        XCTAssertEqual(loss, 0.25, accuracy: 0.001)
    }
    
    func testCrossEntropyBackward() {
        let predicted = Matrix(rows: 3, cols: 1, data: [0.1, 0.2, 0.7])
        let target = Matrix(rows: 3, cols: 1, data: [0.0, 0.0, 1.0])
        
        let grad = Loss.crossEntropyBackward(predicted: predicted, target: target)
        
        // Gradient w.r.t. the correct class should be negative
        XCTAssertTrue(grad.data[2] < 0.0)
        
        // Gradients w.r.t. wrong classes should be zero (since target is 0)
        XCTAssertEqual(grad.data[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(grad.data[1], 0.0, accuracy: 0.001)
    }
}

final class LayerTests: XCTestCase {
    
    func testDenseLayerForward() {
        let layer = DenseLayer(inputSize: 2, outputSize: 3)
        
        // Manually set weights and bias for testing
        layer.weights = Matrix(rows: 3, cols: 2, data: [1.0, 0.0,
                                                          0.0, 1.0,
                                                          1.0, 1.0])
        layer.bias = Matrix(rows: 3, cols: 1, data: [0.5, 0.5, 0.5])
        
        let input = Matrix(rows: 2, cols: 1, data: [2.0, 3.0])
        let output = layer.forward(input)
        
        // Expected output: [2.0 + 0.5, 3.0 + 0.5, 5.0 + 0.5] = [2.5, 3.5, 5.5]
        XCTAssertEqual(output.data[0], 2.5, accuracy: 0.001)
        XCTAssertEqual(output.data[1], 3.5, accuracy: 0.001)
        XCTAssertEqual(output.data[2], 5.5, accuracy: 0.001)
    }
    
    func testReLULayerForward() {
        let layer = ReLULayer()
        let input = Matrix(rows: 3, cols: 1, data: [-1.0, 0.0, 1.0])
        let output = layer.forward(input)
        
        XCTAssertEqual(output.data, [0.0, 0.0, 1.0])
    }
    
    func testSoftmaxLayerForward() {
        let layer = SoftmaxLayer()
        let input = Matrix(rows: 3, cols: 1, data: [1.0, 2.0, 3.0])
        let output = layer.forward(input)
        
        // Sum should be 1
        XCTAssertEqual(output.sum(), 1.0, accuracy: 0.001)
    }
}

final class ModelTests: XCTestCase {
    
    func testSequentialModelForward() {
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 2, outputSize: 2))
        model.add(ReLULayer())
        
        let input = Matrix(rows: 2, cols: 1, data: [1.0, 2.0])
        let output = model.forward(input)
        
        // Output should be 2x1
        XCTAssertEqual(output.rows, 2)
        XCTAssertEqual(output.cols, 1)
    }
    
    func testSequentialModelTraining() {
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 2, outputSize: 3))
        model.add(ReLULayer())
        model.add(DenseLayer(inputSize: 3, outputSize: 2))
        model.add(SoftmaxLayer())
        
        let input = Matrix(rows: 2, cols: 1, data: [0.5, 0.5])
        let target = Matrix(rows: 2, cols: 1, data: [1.0, 0.0])
        
        // Initial loss
        let initialLoss = model.computeLoss(input: input, target: target)
        
        // Train for a few steps
        for _ in 0..<10 {
            model.trainStep(input: input, target: target, learningRate: 0.1)
        }
        
        // Loss should decrease
        let finalLoss = model.computeLoss(input: input, target: target)
        XCTAssertTrue(finalLoss < initialLoss)
    }
}

// MARK: - New Feature Tests

final class NewLayerTests: XCTestCase {
    
    func testSigmoidLayer() {
        let layer = SigmoidLayer()
        let input = Matrix(rows: 3, cols: 1, data: [0.0, -1.0, 1.0])
        let output = layer.forward(input)
        
        // Sigmoid(0) = 0.5
        XCTAssertEqual(output.data[0], 0.5, accuracy: 0.001)
        // Sigmoid should be in range (0, 1)
        XCTAssertTrue(output.data[1] > 0.0 && output.data[1] < 1.0)
        XCTAssertTrue(output.data[2] > 0.0 && output.data[2] < 1.0)
    }
    
    func testTanhLayer() {
        let layer = TanhLayer()
        let input = Matrix(rows: 3, cols: 1, data: [0.0, -1.0, 1.0])
        let output = layer.forward(input)
        
        // Tanh(0) = 0
        XCTAssertEqual(output.data[0], 0.0, accuracy: 0.001)
        // Tanh should be in range (-1, 1)
        XCTAssertTrue(output.data[1] > -1.0 && output.data[1] < 1.0)
        XCTAssertTrue(output.data[2] > -1.0 && output.data[2] < 1.0)
    }
    
    func testDropoutLayerTraining() {
        let layer = DropoutLayer(dropoutRate: 0.5)
        layer.training = true
        
        let input = Matrix(rows: 100, cols: 1, value: 1.0)
        let output = layer.forward(input)
        
        // Some values should be zero (dropped out)
        let zeroCount = output.data.filter { $0 == 0.0 }.count
        XCTAssertTrue(zeroCount > 0, "Dropout should zero out some values")
        
        // Non-zero values should be scaled up
        let nonZeroValues = output.data.filter { $0 > 0.0 }
        if !nonZeroValues.isEmpty {
            // With 0.5 dropout, non-zero values should be scaled by 1/(1-0.5) = 2
            XCTAssertTrue(nonZeroValues.allSatisfy { $0 > 1.0 }, "Non-zero values should be scaled up")
        }
    }
    
    func testDropoutLayerInference() {
        let layer = DropoutLayer(dropoutRate: 0.5)
        layer.training = false
        
        let input = Matrix(rows: 100, cols: 1, value: 1.0)
        let output = layer.forward(input)
        
        // During inference, all values should pass through unchanged
        XCTAssertEqual(output.data, input.data)
    }
    
    func testBatchNormLayerForward() {
        let layer = BatchNormLayer(numFeatures: 3)
        layer.training = true
        
        let input = Matrix(rows: 3, cols: 1, data: [1.0, 2.0, 3.0])
        let output = layer.forward(input)
        
        // Output should have same shape as input
        XCTAssertEqual(output.rows, 3)
        XCTAssertEqual(output.cols, 1)
        
        // Output should not be constant (would indicate the bug)
        let allSame = output.data.allSatisfy { abs($0 - output.data[0]) < 0.0001 }
        XCTAssertFalse(allSame, "BatchNorm output should not be constant for varying inputs")
    }
    
    func testBatchNormLayerInference() {
        let layer = BatchNormLayer(numFeatures: 3)
        layer.training = false
        
        let input = Matrix(rows: 3, cols: 1, data: [1.0, 2.0, 3.0])
        let output = layer.forward(input)
        
        // Output should have same shape as input
        XCTAssertEqual(output.rows, 3)
        XCTAssertEqual(output.cols, 1)
        
        // Output should not be constant
        let allSame = output.data.allSatisfy { abs($0 - output.data[0]) < 0.0001 }
        XCTAssertFalse(allSame, "BatchNorm output should not be constant for varying inputs")
    }
}

final class OptimizerTests: XCTestCase {
    
    func testSGDOptimizer() {
        let optimizer = SGDOptimizer()
        var params = [Matrix(rows: 2, cols: 2, value: 1.0)]
        let grads = [Matrix(rows: 2, cols: 2, value: 0.1)]
        
        optimizer.update(parameters: &params, gradients: grads, learningRate: 0.1)
        
        // After update: param = 1.0 - 0.1 * 0.1 = 0.99
        XCTAssertEqual(params[0].data[0], 0.99, accuracy: 0.001)
    }
    
    func testSGDMomentumOptimizer() {
        let optimizer = SGDMomentumOptimizer(momentum: 0.9)
        var params = [Matrix(rows: 2, cols: 2, value: 1.0)]
        let grads = [Matrix(rows: 2, cols: 2, value: 0.1)]
        
        // First update
        optimizer.update(parameters: &params, gradients: grads, learningRate: 0.1)
        let firstUpdate = params[0].data[0]
        
        // Second update with same gradient should show momentum effect
        optimizer.update(parameters: &params, gradients: grads, learningRate: 0.1)
        let secondUpdate = params[0].data[0]
        
        // Second update should be larger due to momentum accumulation
        XCTAssertTrue(abs(secondUpdate - firstUpdate) > abs(firstUpdate - 1.0))
    }
    
    func testAdamOptimizer() {
        let optimizer = AdamOptimizer()
        var params = [Matrix(rows: 2, cols: 2, value: 1.0)]
        let grads = [Matrix(rows: 2, cols: 2, value: 0.1)]
        
        optimizer.update(parameters: &params, gradients: grads, learningRate: 0.001)
        
        // Parameters should be updated
        XCTAssertNotEqual(params[0].data[0], 1.0)
        XCTAssertTrue(params[0].data[0] < 1.0)
    }
    
    func testRMSpropOptimizer() {
        let optimizer = RMSpropOptimizer()
        var params = [Matrix(rows: 2, cols: 2, value: 1.0)]
        let grads = [Matrix(rows: 2, cols: 2, value: 0.1)]
        
        optimizer.update(parameters: &params, gradients: grads, learningRate: 0.001)
        
        // Parameters should be updated
        XCTAssertNotEqual(params[0].data[0], 1.0)
        XCTAssertTrue(params[0].data[0] < 1.0)
    }
}

final class SerializationTests: XCTestCase {
    
    func testModelSaveAndLoad() throws {
        // Create a model
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 2, outputSize: 3))
        model.add(ReLULayer())
        model.add(DenseLayer(inputSize: 3, outputSize: 2))
        model.add(SoftmaxLayer())
        
        // Save to temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_model.json")
        try model.save(to: tempURL)
        
        // Load the model
        let loadedModel = try SequentialModel.load(from: tempURL)
        
        // Test that loaded model produces same output
        let input = Matrix(rows: 2, cols: 1, data: [0.5, 0.5])
        let originalOutput = model.forward(input)
        let loadedOutput = loadedModel.forward(input)
        
        // Outputs should be very close
        for i in 0..<originalOutput.data.count {
            XCTAssertEqual(originalOutput.data[i], loadedOutput.data[i], accuracy: 0.0001)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testModelSaveWithDifferentLayers() throws {
        // Create a model with various layer types
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 4, outputSize: 8))
        model.add(SigmoidLayer())
        model.add(DenseLayer(inputSize: 8, outputSize: 4))
        model.add(TanhLayer())
        
        // Save to temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_model2.json")
        try model.save(to: tempURL)
        
        // Load the model
        let loadedModel = try SequentialModel.load(from: tempURL)
        
        // Test that loaded model produces same output
        let input = Matrix(rows: 4, cols: 1, data: [0.1, 0.2, 0.3, 0.4])
        let originalOutput = model.forward(input)
        let loadedOutput = loadedModel.forward(input)
        
        // Outputs should be very close
        for i in 0..<originalOutput.data.count {
            XCTAssertEqual(originalOutput.data[i], loadedOutput.data[i], accuracy: 0.0001)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testBatchNormSerialization() throws {
        // Create a model with batch normalization
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 4, outputSize: 8))
        model.add(BatchNormLayer(numFeatures: 8))
        model.add(ReLULayer())
        model.add(DenseLayer(inputSize: 8, outputSize: 2))
        
        // Train for a bit to update running statistics
        let input = Matrix(rows: 4, cols: 1, data: [0.1, 0.2, 0.3, 0.4])
        let target = Matrix(rows: 2, cols: 1, data: [1.0, 0.0])
        for _ in 0..<10 {
            model.trainStep(input: input, target: target, learningRate: 0.01)
        }
        
        // Set to inference mode
        for layer in model.getLayers() {
            if let bn = layer as? BatchNormLayer {
                bn.training = false
            }
        }
        
        // Save to temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_batchnorm.json")
        try model.save(to: tempURL)
        
        // Load the model
        let loadedModel = try SequentialModel.load(from: tempURL)
        
        // Set loaded model to inference mode
        for layer in loadedModel.getLayers() {
            if let bn = layer as? BatchNormLayer {
                bn.training = false
            }
        }
        
        // Test that outputs match in inference mode
        let originalOutput = model.forward(input)
        let loadedOutput = loadedModel.forward(input)
        
        for i in 0..<originalOutput.data.count {
            XCTAssertEqual(originalOutput.data[i], loadedOutput.data[i], accuracy: 0.001)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testDropoutSerialization() throws {
        // Create a model with dropout
        let model = SequentialModel()
        model.add(DenseLayer(inputSize: 4, outputSize: 8))
        model.add(DropoutLayer(dropoutRate: 0.3))
        model.add(ReLULayer())
        model.add(DenseLayer(inputSize: 8, outputSize: 2))
        
        // Save to temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_dropout.json")
        try model.save(to: tempURL)
        
        // Load the model
        let loadedModel = try SequentialModel.load(from: tempURL)
        
        // Set both to inference mode for consistent testing
        for layer in model.getLayers() {
            if let dropout = layer as? DropoutLayer {
                dropout.training = false
            }
        }
        for layer in loadedModel.getLayers() {
            if let dropout = layer as? DropoutLayer {
                dropout.training = false
            }
        }
        
        // Test that outputs match in inference mode
        let input = Matrix(rows: 4, cols: 1, data: [0.1, 0.2, 0.3, 0.4])
        let originalOutput = model.forward(input)
        let loadedOutput = loadedModel.forward(input)
        
        for i in 0..<originalOutput.data.count {
            XCTAssertEqual(originalOutput.data[i], loadedOutput.data[i], accuracy: 0.0001)
        }
        
        // Verify dropout rate was preserved
        var foundDropout = false
        for layer in loadedModel.getLayers() {
            if let dropout = layer as? DropoutLayer {
                XCTAssertEqual(dropout.rate, 0.3, accuracy: 0.001)
                foundDropout = true
            }
        }
        XCTAssertTrue(foundDropout, "Dropout layer should be present in loaded model")
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
}
