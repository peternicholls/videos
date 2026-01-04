/// DataUtilsTests.swift
/// Tests for data loading and preprocessing utilities

import XCTest
@testable import MLSwift

final class DataPreprocessingTests: XCTestCase {
    
    // MARK: - Normalization Tests
    
    func testNormalizeToRange() {
        let matrix = Matrix(rows: 2, cols: 2, data: [0, 127.5, 255, 100])
        let normalized = normalizeToRange(matrix, maxValue: 255.0)
        
        XCTAssertEqual(normalized[0, 0], 0.0, accuracy: 0.001)
        XCTAssertEqual(normalized[0, 1], 0.5, accuracy: 0.001)
        XCTAssertEqual(normalized[1, 0], 1.0, accuracy: 0.001)
        XCTAssertEqual(normalized[1, 1], 100.0/255.0, accuracy: 0.001)
    }
    
    func testMinMaxNormalize() {
        let matrix = Matrix(rows: 2, cols: 2, data: [10, 20, 30, 40])
        let normalized = minMaxNormalize(matrix)
        
        XCTAssertEqual(normalized[0, 0], 0.0, accuracy: 0.001)
        XCTAssertEqual(normalized[1, 1], 1.0, accuracy: 0.001)
        XCTAssertEqual(normalized[0, 1], 1.0/3.0, accuracy: 0.001)
    }
    
    func testMinMaxNormalizeWithConstantValues() {
        // Test edge case: all values are identical
        let matrix = Matrix(rows: 2, cols: 2, data: [5.0, 5.0, 5.0, 5.0])
        let normalized = minMaxNormalize(matrix)
        
        // Should return original matrix when range is 0
        XCTAssertEqual(normalized[0, 0], 5.0, accuracy: 0.001)
        XCTAssertEqual(normalized[1, 1], 5.0, accuracy: 0.001)
    }
    
    func testStandardize() {
        let matrix = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let (standardized, mean, stdDev) = standardize(matrix)
        
        XCTAssertEqual(mean, 2.5, accuracy: 0.001)
        XCTAssertGreaterThan(stdDev, 0)
        
        // Check that standardized data has approximately zero mean
        let standardizedMean = standardized.data.reduce(0, +) / Float(standardized.data.count)
        XCTAssertEqual(standardizedMean, 0.0, accuracy: 0.001)
    }
    
    func testStandardizeWithConstantValues() {
        // Test edge case: all values are identical (zero std dev)
        let matrix = Matrix(rows: 2, cols: 2, data: [3.0, 3.0, 3.0, 3.0])
        let (standardized, mean, stdDev) = standardize(matrix)
        
        XCTAssertEqual(mean, 3.0, accuracy: 0.001)
        XCTAssertEqual(stdDev, 0.0, accuracy: 0.001)
        // Should return original matrix when stdDev is 0
        XCTAssertEqual(standardized[0, 0], 3.0, accuracy: 0.001)
    }
    
    // MARK: - Label Encoding Tests
    
    func testOneHotEncode() {
        let labels = [0, 1, 2, 0]
        let encoded = oneHotEncode(labels: labels, numClasses: 3)
        
        XCTAssertEqual(encoded.count, 4)
        XCTAssertEqual(encoded[0][0, 0], 1.0)
        XCTAssertEqual(encoded[0][1, 0], 0.0)
        XCTAssertEqual(encoded[1][1, 0], 1.0)
        XCTAssertEqual(encoded[2][2, 0], 1.0)
    }
    
    func testOneHotEncodeWithOutOfBoundsLabels() {
        // Test edge case: labels out of bounds
        let labels = [-1, 5, 1]  // -1 and 5 are out of bounds for 3 classes
        let encoded = oneHotEncode(labels: labels, numClasses: 3)
        
        XCTAssertEqual(encoded.count, 3)
        // Out of bounds labels should produce zero vectors
        XCTAssertEqual(encoded[0].data.reduce(0, +), 0.0, accuracy: 0.001)
        XCTAssertEqual(encoded[1].data.reduce(0, +), 0.0, accuracy: 0.001)
        // Valid label should work normally
        XCTAssertEqual(encoded[2][1, 0], 1.0)
    }
    
    func testOneHotDecode() {
        let encoded = [
            Matrix(rows: 3, cols: 1, data: [1.0, 0.0, 0.0]),
            Matrix(rows: 3, cols: 1, data: [0.0, 1.0, 0.0]),
            Matrix(rows: 3, cols: 1, data: [0.0, 0.0, 1.0])
        ]
        let decoded = oneHotDecode(encoded)
        
        XCTAssertEqual(decoded, [0, 1, 2])
    }
    
    // MARK: - Dataset Utilities Tests
    
    func testTrainValidationSplit() {
        let data = Array(0..<100)
        let (train, validation) = trainValidationSplit(data: data, splitRatio: 0.8)
        
        XCTAssertEqual(train.count, 80)
        XCTAssertEqual(validation.count, 20)
        XCTAssertEqual(train.first, 0)
        XCTAssertEqual(validation.last, 99)
    }
    
    func testShuffleDataset() {
        let inputs = [
            Matrix(rows: 2, cols: 1, data: [1, 0]),
            Matrix(rows: 2, cols: 1, data: [2, 0]),
            Matrix(rows: 2, cols: 1, data: [3, 0])
        ]
        let targets = [
            Matrix(rows: 1, cols: 1, data: [10]),
            Matrix(rows: 1, cols: 1, data: [20]),
            Matrix(rows: 1, cols: 1, data: [30])
        ]
        
        let (shuffledInputs, shuffledTargets) = shuffleDataset(inputs: inputs, targets: targets)
        
        XCTAssertEqual(shuffledInputs.count, inputs.count)
        XCTAssertEqual(shuffledTargets.count, targets.count)
        
        // Verify correspondence is maintained
        for i in 0..<shuffledInputs.count {
            let inputValue = shuffledInputs[i][0, 0]
            let targetValue = shuffledTargets[i][0, 0]
            XCTAssertEqual(targetValue, inputValue * 10, accuracy: 0.001)
        }
    }
    
    func testCreateBatches() {
        let inputs = (0..<100).map { i in
            Matrix(rows: 1, cols: 1, data: [Float(i)])
        }
        let targets = (0..<100).map { i in
            Matrix(rows: 1, cols: 1, data: [Float(i * 2)])
        }
        
        let batches = createBatches(inputs: inputs, targets: targets, batchSize: 32)
        
        XCTAssertEqual(batches.count, 4) // 100 / 32 = 3.125, so 4 batches
        XCTAssertEqual(batches[0].inputs.count, 32)
        XCTAssertEqual(batches[3].inputs.count, 4) // Last batch has remainder
    }
    
    // MARK: - Image Processing Tests
    
    func testFlattenImages() {
        let image1 = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let image2 = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        let images = [image1, image2]
        
        let flattened = flattenImages(images)
        
        XCTAssertEqual(flattened.count, 2)
        XCTAssertEqual(flattened[0].rows, 4)
        XCTAssertEqual(flattened[0].cols, 1)
        XCTAssertEqual(flattened[0][0, 0], 1.0)
        XCTAssertEqual(flattened[0][3, 0], 4.0)
    }
    
    func testReshapeToImages() {
        let flat1 = Matrix(rows: 4, cols: 1, data: [1, 2, 3, 4])
        let flat2 = Matrix(rows: 4, cols: 1, data: [5, 6, 7, 8])
        let flattened = [flat1, flat2]
        
        let images = reshapeToImages(flattened, height: 2, width: 2)
        
        XCTAssertEqual(images.count, 2)
        XCTAssertEqual(images[0].rows, 2)
        XCTAssertEqual(images[0].cols, 2)
        XCTAssertEqual(images[0][0, 0], 1.0)
        XCTAssertEqual(images[1][1, 1], 8.0)
    }
}

final class DataLoaderTests: XCTestCase {
    
    func testSaveAndLoadBinaryMatrix() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_matrix_\(UUID().uuidString).mat")
        
        // Create and save a matrix
        let original = Matrix(rows: 3, cols: 4, data: Array(stride(from: 0.0, to: 12.0, by: 1.0)))
        try saveBinaryMatrix(original, to: tempURL)
        
        // Load it back
        let loaded = try loadBinaryMatrix(from: tempURL, rows: 3, cols: 4)
        
        // Verify they match
        XCTAssertEqual(loaded.rows, original.rows)
        XCTAssertEqual(loaded.cols, original.cols)
        for i in 0..<loaded.data.count {
            XCTAssertEqual(loaded.data[i], original.data[i], accuracy: 0.001)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testLoadBinaryMatrixWithWrongDimensions() {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_wrong_dims_\(UUID().uuidString).mat")
        
        // Save a 3x4 matrix
        let original = Matrix(rows: 3, cols: 4, data: Array(stride(from: 0.0, to: 12.0, by: 1.0)))
        try? saveBinaryMatrix(original, to: tempURL)
        
        // Try to load with wrong dimensions
        XCTAssertThrowsError(try loadBinaryMatrix(from: tempURL, rows: 2, cols: 2)) { error in
            if let loaderError = error as? DataLoaderError {
                switch loaderError {
                case .fileSizeMismatch:
                    XCTAssert(true) // Expected error
                default:
                    XCTFail("Wrong error type")
                }
            } else {
                XCTFail("Wrong error type")
            }
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
}
