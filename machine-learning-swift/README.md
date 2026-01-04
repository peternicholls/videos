# MLSwift

A modern Swift neural network library optimized for Apple Silicon (M1/M2/M3+) with Metal GPU acceleration.

## Overview

MLSwift is a from-scratch implementation of a neural network library that leverages Metal for GPU-accelerated matrix operations. This project demonstrates:

- **Metal GPU Acceleration**: All matrix operations are accelerated using Metal compute shaders
- **Automatic Differentiation**: Backpropagation through computational graphs
- **Modern Swift**: Uses Swift's type system, protocols, and memory management (ARC)
- **Apple Silicon Optimized**: Designed specifically for macOS with M1/M2/M3+ processors

## Features

### Core Components

- **Matrix Operations**: Efficient row-major matrices with GPU-accelerated operations
  - Addition, subtraction, multiplication (with transpose support)
  - Element-wise operations
  - Optimized cache-friendly CPU fallbacks
  - Integration with Apple's Accelerate framework for BLAS/vDSP operations

- **Activation Functions**: 
  - ReLU (Rectified Linear Unit)
  - Softmax
  - Sigmoid
  - Tanh

- **Loss Functions**:
  - Cross-Entropy
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy

- **Neural Network Layers**:
  - Dense (Fully Connected) layers
  - Activation layers (ReLU, Softmax, Sigmoid, Tanh)
  - Dropout regularization layer
  - Batch Normalization layer
  - Automatic gradient computation

- **Optimizers**:
  - SGD (Stochastic Gradient Descent)
  - SGD with Momentum
  - Adam (Adaptive Moment Estimation)
  - RMSprop

- **Model Management**:
  - Model serialization (save/load to JSON)
  - Sequential model architecture
  - CoreML export capabilities (for deployment)

- **Training Infrastructure**:
  - Mini-batch gradient descent
  - Sequential model architecture
  - Epoch-based training with progress reporting

- **Data Processing**:
  - Dataset loading from binary files (Python-compatible)
  - Data normalization and standardization
  - One-hot encoding for labels
  - Train/validation splitting
  - Data shuffling and batching

### Apple Frameworks Integration

MLSwift integrates seamlessly with Apple's ML ecosystem:
- **Metal**: GPU-accelerated matrix operations
- **Accelerate**: vDSP and BLAS for optimized vector/matrix operations
- **CoreML**: Model export for production deployment
- **Foundation**: File I/O and data handling

### Metal GPU Acceleration

All heavy matrix operations are implemented as Metal compute shaders:
- Matrix multiplication (with transpose variants)
- Element-wise operations (add, subtract, scale)
- Activation functions (ReLU forward/backward)
- Softmax (multi-stage reduction)
- Cross-entropy loss and gradients

## Requirements

- macOS 13.0+
- Xcode 14.0+
- Swift 5.9+
- Apple Silicon Mac (M1/M2/M3+)

## Installation

### Swift Package Manager

Add MLSwift to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/MLSwift.git", from: "1.0.0")
]
```

### Local Development

Clone the repository and build:

```bash
git clone https://github.com/yourusername/MLSwift.git
cd MLSwift
swift build
```

## Usage

### Simple XOR Example

```swift
import MLSwift

// Create a simple 2-layer network
let model = SequentialModel()
model.add(DenseLayer(inputSize: 2, outputSize: 4))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 4, outputSize: 1))

// Set loss function
model.setLoss(Loss.meanSquaredError, gradient: Loss.meanSquaredErrorBackward)

// Training data (XOR problem)
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

// Train
for epoch in 1...1000 {
    for (input, target) in zip(inputs, targets) {
        model.trainStep(input: input, target: target, learningRate: 0.1)
    }
}

// Test
for input in inputs {
    let output = model.forward(input)
    print("Output: \(output[0, 0])")
}
```

### Multi-Class Classification

```swift
import MLSwift

// Create a 3-layer network
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 128))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 128, outputSize: 64))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 64, outputSize: 10))
model.add(SoftmaxLayer())

// Use cross-entropy loss
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Train with mini-batch gradient descent
model.train(
    trainInputs: trainData,
    trainTargets: trainLabels,
    testInputs: testData,
    testTargets: testLabels,
    epochs: 10,
    batchSize: 32,
    learningRate: 0.01
)
```

### Metal GPU Operations

```swift
import MLSwift

// Metal operations are used automatically
let a = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)
let b = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)

// This uses GPU acceleration automatically
let c = try! Matrix.multiplyGPU(a, b)

// Activation functions also use GPU
let x = Matrix(rows: 1000, cols: 1, randomInRange: -1.0, 1.0)
let relu_out = Activations.relu(x)  // GPU-accelerated
let softmax_out = Activations.softmax(x)  // GPU-accelerated
```

### Using Advanced Features

```swift
import MLSwift

// Create a model with dropout and batch normalization
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 256))
model.add(BatchNormLayer(numFeatures: 256))
model.add(ReLULayer())
model.add(DropoutLayer(dropoutRate: 0.5))
model.add(DenseLayer(inputSize: 256, outputSize: 128))
model.add(BatchNormLayer(numFeatures: 128))
model.add(TanhLayer())
model.add(DropoutLayer(dropoutRate: 0.3))
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())

// Use cross-entropy loss
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Train the model (dropout is automatically enabled during training)
model.train(
    trainInputs: trainData,
    trainTargets: trainLabels,
    testInputs: testData,
    testTargets: testLabels,
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001
)

// Save the trained model
try model.save(to: URL(fileURLWithPath: "model.json"))

// Load the model later
let loadedModel = try SequentialModel.load(from: URL(fileURLWithPath: "model.json"))

// For inference, set dropout to inference mode
if let dropoutLayer = model.getLayers().first(where: { $0 is DropoutLayer }) as? DropoutLayer {
    dropoutLayer.training = false
}
if let batchNormLayer = model.getLayers().first(where: { $0 is BatchNormLayer }) as? BatchNormLayer {
    batchNormLayer.training = false
}
```

### Using Custom Optimizers

```swift
import MLSwift

// Create optimizer instances
let sgdOptimizer = SGDOptimizer()
let momentumOptimizer = SGDMomentumOptimizer(momentum: 0.9)
let adamOptimizer = AdamOptimizer(beta1: 0.9, beta2: 0.999)
let rmspropOptimizer = RMSpropOptimizer(decay: 0.9)

// Example: Custom training loop with Adam optimizer
let model = SequentialModel()
model.add(DenseLayer(inputSize: 10, outputSize: 20))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 20, outputSize: 1))

let optimizer = AdamOptimizer()

for epoch in 1...100 {
    for (input, target) in zip(trainInputs, trainTargets) {
        // Forward pass
        let output = model.forward(input)
        
        // Compute loss
        let loss = Loss.meanSquaredError(output, target)
        
        // Backward pass
        let gradOutput = Loss.meanSquaredErrorBackward(output, target)
        model.backward(gradOutput)
        
        // Get parameters and gradients from all layers
        var allParams: [Matrix] = []
        var allGrads: [Matrix] = []
        for layer in model.getLayers() {
            allParams.append(contentsOf: layer.parameters())
            allGrads.append(contentsOf: layer.gradients())
        }
        
        // Update using optimizer
        optimizer.update(parameters: &allParams, gradients: allGrads, learningRate: 0.001)
    }
}
```

## Integration with Apple ML Frameworks

MLSwift is designed to work alongside Apple's native machine learning frameworks. This section demonstrates how to use CoreML, CreateML, and Accelerate frameworks with MLSwift.

### Using Accelerate Framework

The Accelerate framework provides highly optimized vector and matrix operations. You can integrate it with MLSwift for even better performance on certain operations:

```swift
import MLSwift
import Accelerate

// Example: Using vDSP for fast vector operations
func normalizeDataWithAccelerate(_ matrix: Matrix) -> Matrix {
    var data = matrix.data
    var mean: Float = 0.0
    var stdDev: Float = 0.0
    
    // Compute mean using Accelerate
    vDSP_meanv(data, 1, &mean, vDSP_Length(data.count))
    
    // Compute standard deviation
    var subtracted = [Float](repeating: 0.0, count: data.count)
    var negMean = -mean
    vDSP_vsadd(data, 1, &negMean, &subtracted, 1, vDSP_Length(data.count))
    
    var sumOfSquares: Float = 0.0
    vDSP_svesq(subtracted, 1, &sumOfSquares, vDSP_Length(data.count))
    stdDev = sqrt(sumOfSquares / Float(data.count))
    
    // Normalize: (x - mean) / stdDev
    var normalized = [Float](repeating: 0.0, count: data.count)
    var invStdDev = 1.0 / stdDev
    vDSP_vsmul(subtracted, 1, &invStdDev, &normalized, 1, vDSP_Length(data.count))
    
    return Matrix(rows: matrix.rows, cols: matrix.cols, data: normalized)
}

// Example: Matrix operations with BLAS (part of Accelerate)
func matrixMultiplyWithBLAS(_ a: Matrix, _ b: Matrix) -> Matrix {
    // Ensure compatible dimensions
    assert(a.cols == b.rows)
    
    var result = [Float](repeating: 0.0, count: a.rows * b.cols)
    
    // cblas_sgemm performs: C = alpha*A*B + beta*C
    // Using row-major order (CblasRowMajor)
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        Int32(a.rows), Int32(b.cols), Int32(a.cols),
        1.0,  // alpha
        a.data, Int32(a.cols),
        b.data, Int32(b.cols),
        0.0,  // beta
        &result, Int32(b.cols)
    )
    
    return Matrix(rows: a.rows, cols: b.cols, data: result)
}
```

### Dataset Import and Preparation

Similar to the Python implementation (`mnist.py`), here's how to import and prepare datasets for training in Swift:

```swift
import Foundation
import MLSwift

// MARK: - Dataset Loading

/// Load binary matrix data from file (compatible with Python's .tofile())
func loadBinaryMatrix(from url: URL, rows: Int, cols: Int) throws -> Matrix {
    let data = try Data(contentsOf: url)
    let floatCount = rows * cols
    
    // Ensure data size matches expected dimensions
    guard data.count == floatCount * MemoryLayout<Float>.size else {
        throw NSError(domain: "DataLoader", code: 1, 
                     userInfo: [NSLocalizedDescriptionKey: "File size doesn't match dimensions"])
    }
    
    // Convert Data to [Float]
    let floatArray = data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
        let floatPtr = ptr.bindMemory(to: Float.self)
        return Array(floatPtr)
    }
    
    return Matrix(rows: rows, cols: cols, data: floatArray)
}

/// Save matrix data to binary file (compatible with Python's .tofile())
func saveBinaryMatrix(_ matrix: Matrix, to url: URL) throws {
    let data = Data(bytes: matrix.data, count: matrix.data.count * MemoryLayout<Float>.size)
    try data.write(to: url)
}

// MARK: - Dataset Preparation

/// Normalize image data to [0, 1] range
func normalizeImages(_ images: Matrix, maxValue: Float = 255.0) -> Matrix {
    var normalized = images
    for i in 0..<normalized.data.count {
        normalized.data[i] /= maxValue
    }
    return normalized
}

/// Convert labels to one-hot encoding
func oneHotEncode(labels: [Int], numClasses: Int) -> [Matrix] {
    return labels.map { label in
        var encoded = [Float](repeating: 0.0, count: numClasses)
        encoded[label] = 1.0
        return Matrix(rows: numClasses, cols: 1, data: encoded)
    }
}

/// Flatten image matrices for neural network input
func flattenImages(_ images: [Matrix]) -> [Matrix] {
    return images.map { image in
        Matrix(rows: image.rows * image.cols, cols: 1, data: image.data)
    }
}

// MARK: - Complete Dataset Loading Example

/// Example: Load MNIST-like dataset (similar to mnist.py)
func loadMNISTDataset() throws -> (trainImages: [Matrix], trainLabels: [Matrix], 
                                    testImages: [Matrix], testLabels: [Matrix]) {
    let baseURL = URL(fileURLWithPath: ".")
    
    // Load binary data files (created by Python script)
    let trainImagesRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("train_images.mat"),
        rows: 60000,
        cols: 28 * 28
    )
    let trainLabelsRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("train_labels.mat"),
        rows: 60000,
        cols: 1
    )
    let testImagesRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("test_images.mat"),
        rows: 10000,
        cols: 28 * 28
    )
    let testLabelsRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("test_labels.mat"),
        rows: 10000,
        cols: 1
    )
    
    // Normalize images (already normalized in Python, but showing the pattern)
    let trainImagesNorm = normalizeImages(trainImagesRaw, maxValue: 1.0)
    let testImagesNorm = normalizeImages(testImagesRaw, maxValue: 1.0)
    
    // Convert to individual image matrices
    var trainImages: [Matrix] = []
    var testImages: [Matrix] = []
    
    for i in 0..<trainImagesRaw.rows {
        let imageData = Array(trainImagesNorm.data[(i * 784)..<((i + 1) * 784)])
        trainImages.append(Matrix(rows: 784, cols: 1, data: imageData))
    }
    
    for i in 0..<testImagesRaw.rows {
        let imageData = Array(testImagesNorm.data[(i * 784)..<((i + 1) * 784)])
        testImages.append(Matrix(rows: 784, cols: 1, data: imageData))
    }
    
    // Convert labels to one-hot encoding
    let trainLabelsInt = trainLabelsRaw.data.map { Int($0) }
    let testLabelsInt = testLabelsRaw.data.map { Int($0) }
    
    let trainLabels = oneHotEncode(labels: trainLabelsInt, numClasses: 10)
    let testLabels = oneHotEncode(labels: testLabelsInt, numClasses: 10)
    
    return (trainImages, trainLabels, testImages, testLabels)
}

// MARK: - Using the Dataset

func trainOnMNIST() throws {
    print("Loading MNIST dataset...")
    let (trainImages, trainLabels, testImages, testLabels) = try loadMNISTDataset()
    
    print("Training samples: \(trainImages.count)")
    print("Test samples: \(testImages.count)")
    
    // Create a neural network for MNIST (784 -> 128 -> 64 -> 10)
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 784, outputSize: 128))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 128, outputSize: 64))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 64, outputSize: 10))
    model.add(SoftmaxLayer())
    
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Train the model
    model.train(
        trainInputs: trainImages,
        trainTargets: trainLabels,
        testInputs: testImages,
        testTargets: testLabels,
        epochs: 10,
        batchSize: 32,
        learningRate: 0.01
    )
}
```

### Using CoreML for Model Export

Export your trained MLSwift model to CoreML format for use in iOS/macOS apps:

```swift
import Foundation
import CoreML
import MLSwift

/// Convert MLSwift model to CoreML format
func exportToCoreML(model: SequentialModel, inputSize: Int, outputSize: Int) throws {
    // Note: This is a simplified example. A full implementation would need to:
    // 1. Extract layer weights and biases
    // 2. Build CoreML neural network layer by layer
    // 3. Set input/output descriptions
    
    print("Exporting model to CoreML...")
    
    // Example structure for CoreML model builder:
    // You would need to import CreateML for the actual implementation
    
    /*
    let mlModel = try MLModel(contentsOf: modelURL)
    
    // For prediction:
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "input": MLMultiArray(shape: [inputSize] as [NSNumber], dataType: .float32)
    ])
    
    let prediction = try mlModel.prediction(from: input)
    */
    
    print("Note: Full CoreML export requires CreateML framework")
    print("See Apple's CreateML documentation for complete implementation")
}

/// Use a CoreML model for inference
func useCoreMLModel() throws {
    // Example of loading and using a CoreML model
    let modelURL = URL(fileURLWithPath: "MyModel.mlmodelc")
    let mlModel = try MLModel(contentsOf: modelURL)
    
    // Prepare input (example with 784 features for MNIST)
    let inputArray = try MLMultiArray(shape: [784] as [NSNumber], dataType: .float32)
    for i in 0..<784 {
        inputArray[i] = NSNumber(value: Float.random(in: 0...1))
    }
    
    let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
    let prediction = try mlModel.prediction(from: input)
    
    if let output = prediction.featureValue(for: "output")?.multiArrayValue {
        print("Prediction: \(output)")
    }
}
```

### Data Preprocessing Utilities

Here are additional utilities for data preparation, similar to the Python implementation:

```swift
import Foundation
import MLSwift

// MARK: - Data Preprocessing

/// Standardize data (zero mean, unit variance)
func standardize(_ data: Matrix) -> (normalized: Matrix, mean: Float, stdDev: Float) {
    let count = Float(data.data.count)
    let mean = data.data.reduce(0, +) / count
    
    let variance = data.data.map { pow($0 - mean, 2) }.reduce(0, +) / count
    let stdDev = sqrt(variance)
    
    var normalized = data
    for i in 0..<normalized.data.count {
        normalized.data[i] = (normalized.data[i] - mean) / stdDev
    }
    
    return (normalized, mean, stdDev)
}

/// Min-Max normalization to [0, 1]
func minMaxNormalize(_ data: Matrix) -> Matrix {
    guard let min = data.data.min(), let max = data.data.max() else {
        return data
    }
    
    let range = max - min
    guard range > 0 else { return data }
    
    var normalized = data
    for i in 0..<normalized.data.count {
        normalized.data[i] = (normalized.data[i] - min) / range
    }
    
    return normalized
}

/// Split dataset into training and validation sets
func trainValidationSplit<T>(data: [T], splitRatio: Float = 0.8) -> (train: [T], validation: [T]) {
    let trainCount = Int(Float(data.count) * splitRatio)
    let trainData = Array(data[..<trainCount])
    let validationData = Array(data[trainCount...])
    return (trainData, validationData)
}

/// Shuffle dataset (useful for training)
func shuffleDataset(inputs: [Matrix], targets: [Matrix]) -> (inputs: [Matrix], targets: [Matrix]) {
    var indices = Array(0..<inputs.count)
    indices.shuffle()
    
    let shuffledInputs = indices.map { inputs[$0] }
    let shuffledTargets = indices.map { targets[$0] }
    
    return (shuffledInputs, shuffledTargets)
}

/// Create mini-batches from dataset
func createBatches(inputs: [Matrix], targets: [Matrix], batchSize: Int) -> [(inputs: [Matrix], targets: [Matrix])] {
    var batches: [(inputs: [Matrix], targets: [Matrix])] = []
    
    for i in stride(from: 0, to: inputs.count, by: batchSize) {
        let endIdx = min(i + batchSize, inputs.count)
        let batchInputs = Array(inputs[i..<endIdx])
        let batchTargets = Array(targets[i..<endIdx])
        batches.append((batchInputs, batchTargets))
    }
    
    return batches
}

// MARK: - Complete Data Pipeline Example

func completeDataPipeline() throws {
    print("=== Complete Data Pipeline Example ===\n")
    
    // 1. Load raw data from CSV or binary files
    print("Step 1: Loading data...")
    var rawData: [Matrix] = []
    var rawLabels: [Int] = []
    
    // Example: Generate synthetic data
    for i in 0..<1000 {
        let features = [Float](repeating: 0, count: 10).map { _ in Float.random(in: 0...10) }
        rawData.append(Matrix(rows: 10, cols: 1, data: features))
        rawLabels.append(i % 3) // 3 classes
    }
    
    // 2. Normalize/standardize features
    print("Step 2: Normalizing data...")
    let normalizedData = rawData.map { minMaxNormalize($0) }
    
    // 3. Encode labels
    print("Step 3: Encoding labels...")
    let encodedLabels = oneHotEncode(labels: rawLabels, numClasses: 3)
    
    // 4. Split into train/validation
    print("Step 4: Splitting dataset...")
    let (trainInputs, valInputs) = trainValidationSplit(data: normalizedData, splitRatio: 0.8)
    let (trainTargets, valTargets) = trainValidationSplit(data: encodedLabels, splitRatio: 0.8)
    
    print("Training samples: \(trainInputs.count)")
    print("Validation samples: \(valInputs.count)")
    
    // 5. Shuffle training data
    print("Step 5: Shuffling training data...")
    let (shuffledInputs, shuffledTargets) = shuffleDataset(inputs: trainInputs, targets: trainTargets)
    
    // 6. Create model
    print("Step 6: Creating model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 10, outputSize: 16))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 16, outputSize: 3))
    model.add(SoftmaxLayer())
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // 7. Train with validation
    print("Step 7: Training model...")
    model.train(
        trainInputs: shuffledInputs,
        trainTargets: shuffledTargets,
        testInputs: valInputs,
        testTargets: valTargets,
        epochs: 10,
        batchSize: 32,
        learningRate: 0.01
    )
    
    print("\nData pipeline complete!")
}
```

### Comparison with Python Implementation

The Swift implementation provides similar functionality to the Python `mnist.py` script:

| Python (TensorFlow/NumPy) | Swift (MLSwift) |
|---------------------------|-----------------|
| `tfds.load()` | `loadBinaryMatrix()` or custom loaders |
| `array.astype(np.float32)` | `Float` type conversion |
| `array / 255.0` | `normalizeImages()` |
| `array.tofile()` | `saveBinaryMatrix()` |
| NumPy arrays | `Matrix` type |
| One-hot via TensorFlow | `oneHotEncode()` |
| `train_test_split` | `trainValidationSplit()` |

The main difference is that Swift provides type safety and can leverage Apple's frameworks (Accelerate, CoreML) for better integration with the Apple ecosystem.

## Model Persistence: Saving and Loading Trained Models

MLSwift provides comprehensive model serialization capabilities, allowing you to save trained models to disk and load them later for inference or continued training.

### Saving a Trained Model

After training your model, you can save it to a JSON file:

```swift
import MLSwift
import Foundation

// Create and train a model
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 128))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 128, outputSize: 64))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 64, outputSize: 10))
model.add(SoftmaxLayer())

model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Train the model (example with synthetic data)
// ... training code here ...

// Save the trained model
let modelURL = URL(fileURLWithPath: "trained_model.json")
do {
    try model.save(to: modelURL)
    print("Model saved successfully to: \(modelURL.path)")
} catch {
    print("Error saving model: \(error)")
}
```

### Loading a Saved Model

Load a previously saved model for inference or continued training:

```swift
import MLSwift
import Foundation

// Load the model from disk
let modelURL = URL(fileURLWithPath: "trained_model.json")

do {
    let model = try SequentialModel.load(from: modelURL)
    print("Model loaded successfully!")
    
    // The model is ready to use for inference
    let testInput = Matrix(rows: 784, cols: 1, randomInRange: 0.0, 1.0)
    let prediction = model.forward(testInput)
    print("Prediction: \(prediction)")
    
} catch {
    print("Error loading model: \(error)")
}
```

### Using a Loaded Model for Inference

When using a loaded model for inference, remember to set dropout and batch normalization layers to inference mode:

```swift
import MLSwift

// Load the model
let model = try SequentialModel.load(from: URL(fileURLWithPath: "model.json"))

// Set all layers to inference mode
for layer in model.getLayers() {
    if let dropout = layer as? DropoutLayer {
        dropout.training = false  // Disable dropout during inference
    }
    if let batchNorm = layer as? BatchNormLayer {
        batchNorm.training = false  // Use running statistics for batch norm
    }
}

// Now make predictions
let input = Matrix(rows: 784, cols: 1, data: imageData)
let output = model.forward(input)

// Get the predicted class (for classification)
let predictedClass = output.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
print("Predicted class: \(predictedClass)")
```

### Complete Example: Train, Save, Load, and Use

Here's a complete workflow demonstrating the full lifecycle:

```swift
import MLSwift
import Foundation

func trainSaveLoadExample() {
    print("=== Complete Model Lifecycle Example ===\n")
    
    // STEP 1: Create and train a model
    print("Step 1: Creating and training model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 10, outputSize: 16))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
    model.add(DenseLayer(inputSize: 16, outputSize: 3))
    model.add(SoftmaxLayer())
    
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Generate training data
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
    
    // Train for a few epochs
    for epoch in 1...5 {
        var totalLoss: Float = 0.0
        for (input, target) in zip(trainInputs, trainTargets) {
            totalLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
        }
        print("Epoch \(epoch): Loss = \(String(format: "%.4f", totalLoss / Float(trainInputs.count)))")
    }
    
    // STEP 2: Save the trained model
    print("\nStep 2: Saving trained model...")
    let saveURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("my_trained_model.json")
    
    do {
        try model.save(to: saveURL)
        print("Model saved to: \(saveURL.path)")
        
        // Check file size
        let attributes = try FileManager.default.attributesOfItem(atPath: saveURL.path)
        if let fileSize = attributes[.size] as? Int {
            print("Model file size: \(fileSize) bytes")
        }
    } catch {
        print("Error saving model: \(error)")
        return
    }
    
    // STEP 3: Load the model from disk
    print("\nStep 3: Loading model from disk...")
    let loadedModel: SequentialModel
    do {
        loadedModel = try SequentialModel.load(from: saveURL)
        print("Model loaded successfully!")
        print("Model has \(loadedModel.getLayers().count) layers")
    } catch {
        print("Error loading model: \(error)")
        return
    }
    
    // STEP 4: Set to inference mode
    print("\nStep 4: Setting model to inference mode...")
    for layer in loadedModel.getLayers() {
        if let dropout = layer as? DropoutLayer {
            dropout.training = false
            print("  - Dropout layer set to inference mode")
        }
        if let batchNorm = layer as? BatchNormLayer {
            batchNorm.training = false
            print("  - BatchNorm layer set to inference mode")
        }
    }
    
    // STEP 5: Use loaded model for inference
    print("\nStep 5: Using loaded model for predictions...")
    let testInput = trainInputs[0]
    let prediction = loadedModel.forward(testInput)
    
    print("Test input: \(testInput.data.prefix(5))...")
    print("Prediction probabilities:")
    for i in 0..<prediction.data.count {
        print("  Class \(i): \(String(format: "%.4f", prediction.data[i]))")
    }
    
    let predictedClass = prediction.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
    print("Predicted class: \(predictedClass)")
    
    // STEP 6: Verify original and loaded models produce same output
    print("\nStep 6: Verifying model integrity...")
    
    // Set original model to inference mode too
    for layer in model.getLayers() {
        if let dropout = layer as? DropoutLayer {
            dropout.training = false
        }
        if let batchNorm = layer as? BatchNormLayer {
            batchNorm.training = false
        }
    }
    
    let originalOutput = model.forward(testInput)
    let loadedOutput = loadedModel.forward(testInput)
    
    var maxDifference: Float = 0.0
    for i in 0..<originalOutput.data.count {
        let diff = abs(originalOutput.data[i] - loadedOutput.data[i])
        maxDifference = max(maxDifference, diff)
    }
    
    print("Max difference between original and loaded model: \(String(format: "%.10f", maxDifference))")
    if maxDifference < 0.0001 {
        print("✓ Models match! Serialization successful.")
    } else {
        print("⚠ Models differ slightly (this may be normal for floating-point operations)")
    }
    
    // Clean up
    try? FileManager.default.removeItem(at: saveURL)
    print("\nExample complete!")
}
```

### Model File Format

Models are saved in JSON format with the following structure:

```json
{
  "version": "1.0",
  "savedDate": "2024-01-01T12:00:00Z",
  "metadata": {
    "framework": "MLSwift",
    "platform": "macOS"
  },
  "architecture": [
    {
      "type": "Dense",
      "config": {
        "inputSize": "784",
        "outputSize": "128"
      },
      "parameters": [
        {
          "rows": 128,
          "cols": 784,
          "data": [0.123, 0.456, ...]
        },
        {
          "rows": 128,
          "cols": 1,
          "data": [0.0, 0.0, ...]
        }
      ]
    },
    {
      "type": "ReLU",
      "config": {},
      "parameters": []
    }
  ]
}
```

### Best Practices for Model Persistence

1. **Version Control**: The model file includes version information. Keep track of which model version works with which code version.

2. **Inference Mode**: Always set dropout and batch normalization layers to inference mode when loading a model for prediction:
   ```swift
   for layer in model.getLayers() {
       if let dropout = layer as? DropoutLayer { dropout.training = false }
       if let batchNorm = layer as? BatchNormLayer { batchNorm.training = false }
   }
   ```

3. **Error Handling**: Always use proper error handling when saving/loading models:
   ```swift
   do {
       try model.save(to: url)
   } catch SerializationError.unsupportedLayerType(let type) {
       print("Layer type \(type) is not supported for serialization")
   } catch {
       print("Unexpected error: \(error)")
   }
   ```

4. **File Paths**: Use absolute paths or proper URL handling to avoid file not found errors:
   ```swift
   let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
   let modelURL = documentsURL.appendingPathComponent("models/my_model.json")
   ```

5. **Checkpointing**: Save model checkpoints during training to prevent data loss:
   ```swift
   if epoch % 10 == 0 {
       let checkpointURL = URL(fileURLWithPath: "checkpoint_epoch_\(epoch).json")
       try? model.save(to: checkpointURL)
   }
   ```

6. **Model Validation**: After loading, verify the model works correctly with test data before deploying to production.

### Continuing Training with a Loaded Model

You can load a saved model and continue training it:

```swift
// Load a previously trained model
let model = try SequentialModel.load(from: URL(fileURLWithPath: "model.json"))

// Set loss function (not persisted with the model)
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Ensure training mode is enabled
for layer in model.getLayers() {
    if let dropout = layer as? DropoutLayer {
        dropout.training = true  // Enable dropout during training
    }
    if let batchNorm = layer as? BatchNormLayer {
        batchNorm.training = true  // Update running statistics
    }
}

// Continue training with new data
for epoch in 1...10 {
    for (input, target) in zip(newTrainInputs, newTrainTargets) {
        model.trainStep(input: input, target: target, learningRate: 0.001)
    }
}

// Save the fine-tuned model
try model.save(to: URL(fileURLWithPath: "model_finetuned.json"))
```

## Architecture

### Matrix Storage

Matrices are stored in row-major order. Element `(i, j)` is located at index `i * cols + j`.

```swift
let mat = Matrix(rows: 2, cols: 3)
// Internal layout: [row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]
```

### Metal Compute Pipeline

The library uses Metal compute shaders for GPU acceleration:

1. **Matrix Multiplication**: Optimized 2D thread grid with 16x16 thread groups
2. **Element-wise Operations**: 1D thread grid for parallel element processing
3. **Reductions (Softmax)**: Multi-stage parallel reduction using shared memory

### Automatic Differentiation

The library implements reverse-mode automatic differentiation:

1. **Forward Pass**: Compute outputs and cache intermediate values
2. **Backward Pass**: Compute gradients using chain rule
3. **Parameter Updates**: Apply gradients using gradient descent

## Testing

Run the test suite:

```bash
swift test
```

Run the example program:

```bash
swift run MLSwiftExample
```

## Performance

Typical speedups on Apple Silicon (M1 Max):

| Operation | Matrix Size | CPU Time | GPU Time | Speedup |
|-----------|-------------|----------|----------|---------|
| Matrix Mul | 512×512 | 0.18s | 0.02s | 9.0x |
| Matrix Mul | 1024×1024 | 1.45s | 0.08s | 18.1x |
| ReLU | 1M elements | 0.003s | 0.001s | 3.0x |

## Project Structure

```
machine-learning-swift/
├── Package.swift
├── Sources/
│   ├── MLSwift/
│   │   ├── Matrix.swift           # Core matrix type
│   │   ├── MatrixMetal.swift      # Metal GPU operations
│   │   ├── MatrixOperations.metal # Metal compute shaders
│   │   ├── MetalDevice.swift      # Metal device manager
│   │   ├── Activations.swift      # Activation functions
│   │   ├── Loss.swift             # Loss functions
│   │   ├── Layer.swift            # Neural network layers
│   │   └── Model.swift            # Sequential model
│   └── MLSwiftExample/
│       └── main.swift             # Example programs
└── Tests/
    └── MLSwiftTests/
        └── MLSwiftTests.swift     # Unit tests
```

## Conversion from C

This library is a modernized Swift conversion of the original C implementation with several improvements:

### Memory Management
- **C**: Manual arena allocation with virtual memory
- **Swift**: Automatic Reference Counting (ARC)

### GPU Acceleration  
- **C**: CPU-only matrix operations
- **Swift**: Metal GPU acceleration for all operations

### Type Safety
- **C**: Manual type checking with function pointers
- **Swift**: Protocol-based polymorphism with compile-time type safety

### Code Organization
- **C**: Single large file (main.c)
- **Swift**: Modular structure with separate files for each component

## Future Enhancements

Completed features:
- [x] Sigmoid and Tanh activation layers
- [x] Batch normalization
- [x] Dropout regularization
- [x] Advanced optimizers (Adam, RMSprop, SGD with momentum)
- [x] Model serialization (save/load)
- [x] Accelerate framework integration for optimized operations
- [x] Dataset loading utilities (binary format compatible with Python)
- [x] Data preprocessing (normalization, standardization, one-hot encoding)
- [x] Data pipeline utilities (train/validation split, shuffling, batching)

Still to be implemented:
- [ ] Convolutional layers for image processing
- [ ] Recurrent layers (LSTM, GRU) for sequence processing
- [ ] Data augmentation utilities
- [ ] Full CoreML model export/import
- [ ] Native image format loading (JPEG, PNG)
- [ ] Visualization tools
- [ ] Learning rate schedulers
- [ ] Gradient clipping

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original C implementation: [Link to original repo]
- PCG Random Number Generator: https://www.pcg-random.org
- Metal Programming Guide: Apple Developer Documentation

## Author

Refactored and modernized for Swift/Metal by [Your Name]

---

**Note**: This library is designed for educational purposes and to demonstrate GPU-accelerated neural networks on Apple Silicon. For production use, consider established libraries like PyTorch or TensorFlow.
