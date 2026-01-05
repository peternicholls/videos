# Part 6: Data Processing

This part covers loading datasets, preprocessing data, and building complete data pipelines with MLSwift.

## Table of Contents

1. [Loading Data](#loading-data)
2. [Data Preprocessing](#data-preprocessing)
3. [One-Hot Encoding](#one-hot-encoding)
4. [Data Pipeline](#data-pipeline)
5. [Working with MNIST](#working-with-mnist)
6. [Next Steps](#next-steps)

---

## Loading Data

### Loading Binary Files

MLSwift can load binary matrix files (compatible with Python's `numpy.tofile()`):

```swift
import Foundation
import MLSwift

func loadBinaryMatrix(from url: URL, rows: Int, cols: Int) throws -> Matrix {
    let data = try Data(contentsOf: url)
    let expectedSize = rows * cols * MemoryLayout<Float>.size
    
    guard data.count == expectedSize else {
        throw NSError(domain: "DataLoader", code: 1,
                     userInfo: [NSLocalizedDescriptionKey: "File size mismatch"])
    }
    
    let floatArray = data.withUnsafeBytes { buffer -> [Float] in
        let floatBuffer = buffer.bindMemory(to: Float.self)
        return Array(floatBuffer)
    }
    
    return Matrix(rows: rows, cols: cols, data: floatArray)
}

// Usage
let images = try loadBinaryMatrix(
    from: URL(fileURLWithPath: "train_images.bin"),
    rows: 60000,
    cols: 784
)
```

### Saving Binary Files

```swift
func saveBinaryMatrix(_ matrix: Matrix, to url: URL) throws {
    let data = Data(bytes: matrix.data, count: matrix.data.count * MemoryLayout<Float>.size)
    try data.write(to: url)
}

// Usage
try saveBinaryMatrix(processedData, to: URL(fileURLWithPath: "processed.bin"))
```

### Loading CSV Files

```swift
func loadCSV(from url: URL) throws -> (data: [[Float]], headers: [String]?) {
    let content = try String(contentsOf: url, encoding: .utf8)
    let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
    
    guard !lines.isEmpty else { return ([], nil) }
    
    // Check if first line is header
    let firstLine = lines[0].components(separatedBy: ",")
    let hasHeader = firstLine.allSatisfy { Float($0) == nil }
    
    let headers: [String]? = hasHeader ? firstLine : nil
    let dataLines = hasHeader ? Array(lines.dropFirst()) : lines
    
    let data = dataLines.map { line -> [Float] in
        line.components(separatedBy: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
    }
    
    return (data, headers)
}

// Usage
let (data, headers) = try loadCSV(from: URL(fileURLWithPath: "dataset.csv"))
let features = data.map { row in
    Matrix(rows: row.count - 1, cols: 1, data: Array(row.dropLast()))
}
let labels = data.map { Int($0.last!) }
```

## Data Preprocessing

### Normalization (0 to 1)

Scale values to [0, 1] range:

```swift
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

// Usage: Normalize image pixels from [0, 255] to [0, 1]
let normalizedImages = minMaxNormalize(rawImages)
```

### Standardization (Zero Mean, Unit Variance)

Transform to mean=0, std=1:

```swift
func standardize(_ data: Matrix) -> (normalized: Matrix, mean: Float, stdDev: Float) {
    let count = Float(data.data.count)
    let mean = data.data.reduce(0, +) / count
    
    let variance = data.data.map { pow($0 - mean, 2) }.reduce(0, +) / count
    let stdDev = sqrt(variance)
    
    guard stdDev > 0 else { return (data, mean, 0) }
    
    var normalized = data
    for i in 0..<normalized.data.count {
        normalized.data[i] = (normalized.data[i] - mean) / stdDev
    }
    
    return (normalized, mean, stdDev)
}

// Usage
let (standardizedData, mean, std) = standardize(features)

// Apply same transformation to test data
func applyStandardization(_ data: Matrix, mean: Float, stdDev: Float) -> Matrix {
    var result = data
    for i in 0..<result.data.count {
        result.data[i] = (result.data[i] - mean) / stdDev
    }
    return result
}
```

### Per-Feature Normalization

For datasets with features on different scales:

```swift
func normalizeFeatures(_ samples: [Matrix]) -> (normalized: [Matrix], means: [Float], stds: [Float]) {
    guard let first = samples.first else { return ([], [], []) }
    
    let numFeatures = first.data.count
    var means = [Float](repeating: 0, count: numFeatures)
    var stds = [Float](repeating: 0, count: numFeatures)
    
    // Compute per-feature mean
    for sample in samples {
        for i in 0..<numFeatures {
            means[i] += sample.data[i]
        }
    }
    means = means.map { $0 / Float(samples.count) }
    
    // Compute per-feature std
    for sample in samples {
        for i in 0..<numFeatures {
            stds[i] += pow(sample.data[i] - means[i], 2)
        }
    }
    stds = stds.map { sqrt($0 / Float(samples.count)) }
    
    // Normalize
    let normalized = samples.map { sample -> Matrix in
        var result = sample
        for i in 0..<numFeatures {
            if stds[i] > 0 {
                result.data[i] = (result.data[i] - means[i]) / stds[i]
            }
        }
        return result
    }
    
    return (normalized, means, stds)
}
```

## One-Hot Encoding

Convert integer labels to one-hot vectors:

```swift
func oneHotEncode(labels: [Int], numClasses: Int) -> [Matrix] {
    return labels.map { label in
        var encoded = [Float](repeating: 0.0, count: numClasses)
        encoded[label] = 1.0
        return Matrix(rows: numClasses, cols: 1, data: encoded)
    }
}

// Usage
let labels = [0, 1, 2, 1, 0, 2]
let oneHot = oneHotEncode(labels: labels, numClasses: 3)

// Result:
// labels[0]=0 → [1, 0, 0]
// labels[1]=1 → [0, 1, 0]
// labels[2]=2 → [0, 0, 1]
```

### Decoding One-Hot

```swift
func decodeOneHot(_ matrix: Matrix) -> Int {
    return matrix.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
}

// Usage
let prediction = model.forward(input)
let predictedClass = decodeOneHot(prediction)
```

## Data Pipeline

### Train/Validation Split

```swift
func trainValidationSplit<T>(data: [T], splitRatio: Float = 0.8) -> (train: [T], validation: [T]) {
    let trainCount = Int(Float(data.count) * splitRatio)
    return (Array(data[..<trainCount]), Array(data[trainCount...]))
}

// Usage
let (trainInputs, valInputs) = trainValidationSplit(data: allInputs, splitRatio: 0.8)
let (trainTargets, valTargets) = trainValidationSplit(data: allTargets, splitRatio: 0.8)
```

### Shuffling

```swift
func shuffleDataset(inputs: [Matrix], targets: [Matrix]) -> (inputs: [Matrix], targets: [Matrix]) {
    var indices = Array(0..<inputs.count)
    indices.shuffle()
    
    return (indices.map { inputs[$0] }, indices.map { targets[$0] })
}

// Usage: Shuffle before each epoch
for epoch in 1...epochs {
    let (shuffledInputs, shuffledTargets) = shuffleDataset(inputs: trainInputs, targets: trainTargets)
    // Train on shuffled data...
}
```

### Creating Batches

```swift
func createBatches(inputs: [Matrix], targets: [Matrix], batchSize: Int) -> [(inputs: [Matrix], targets: [Matrix])] {
    var batches: [(inputs: [Matrix], targets: [Matrix])] = []
    
    for i in stride(from: 0, to: inputs.count, by: batchSize) {
        let endIdx = min(i + batchSize, inputs.count)
        batches.append((Array(inputs[i..<endIdx]), Array(targets[i..<endIdx])))
    }
    
    return batches
}

// Usage
let batches = createBatches(inputs: trainInputs, targets: trainTargets, batchSize: 32)
for batch in batches {
    for (input, target) in zip(batch.inputs, batch.targets) {
        model.trainStep(input: input, target: target, learningRate: 0.01)
    }
}
```

### Complete Pipeline

```swift
func prepareDataPipeline(
    rawInputs: [Matrix],
    rawLabels: [Int],
    numClasses: Int,
    splitRatio: Float = 0.8
) -> (trainInputs: [Matrix], trainTargets: [Matrix], 
      valInputs: [Matrix], valTargets: [Matrix],
      means: [Float], stds: [Float]) {
    
    // 1. Normalize features
    let (normalizedInputs, means, stds) = normalizeFeatures(rawInputs)
    
    // 2. One-hot encode labels
    let oneHotTargets = oneHotEncode(labels: rawLabels, numClasses: numClasses)
    
    // 3. Split into train/validation
    let (trainInputs, valInputs) = trainValidationSplit(data: normalizedInputs, splitRatio: splitRatio)
    let (trainTargets, valTargets) = trainValidationSplit(data: oneHotTargets, splitRatio: splitRatio)
    
    // 4. Shuffle training data
    let (shuffledTrainInputs, shuffledTrainTargets) = shuffleDataset(inputs: trainInputs, targets: trainTargets)
    
    return (shuffledTrainInputs, shuffledTrainTargets, valInputs, valTargets, means, stds)
}
```

## Working with MNIST

### Python Preparation Script

First, prepare MNIST data with Python:

```python
# mnist_prepare.py
import numpy as np
import tensorflow_datasets as tfds

# Load MNIST
ds = tfds.load('mnist', as_supervised=True)
train_ds = list(tfds.as_numpy(ds['train']))
test_ds = list(tfds.as_numpy(ds['test']))

# Extract images and labels
train_images = np.array([x[0].flatten() / 255.0 for x in train_ds]).astype(np.float32)
train_labels = np.array([x[1] for x in train_ds]).astype(np.float32)
test_images = np.array([x[0].flatten() / 255.0 for x in test_ds]).astype(np.float32)
test_labels = np.array([x[1] for x in test_ds]).astype(np.float32)

# Save as binary files
train_images.tofile('train_images.bin')
train_labels.tofile('train_labels.bin')
test_images.tofile('test_images.bin')
test_labels.tofile('test_labels.bin')

print(f"Train: {train_images.shape}, Test: {test_images.shape}")
```

### Loading MNIST in Swift

```swift
import Foundation
import MLSwift

func loadMNIST() throws -> (trainImages: [Matrix], trainLabels: [Matrix],
                            testImages: [Matrix], testLabels: [Matrix]) {
    let baseURL = URL(fileURLWithPath: ".")
    
    // Load binary data
    let trainImagesRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("train_images.bin"),
        rows: 60000, cols: 784
    )
    let trainLabelsRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("train_labels.bin"),
        rows: 60000, cols: 1
    )
    let testImagesRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("test_images.bin"),
        rows: 10000, cols: 784
    )
    let testLabelsRaw = try loadBinaryMatrix(
        from: baseURL.appendingPathComponent("test_labels.bin"),
        rows: 10000, cols: 1
    )
    
    // Convert to individual matrices
    var trainImages: [Matrix] = []
    var testImages: [Matrix] = []
    
    for i in 0..<60000 {
        let imageData = Array(trainImagesRaw.data[(i * 784)..<((i + 1) * 784)])
        trainImages.append(Matrix(rows: 784, cols: 1, data: imageData))
    }
    
    for i in 0..<10000 {
        let imageData = Array(testImagesRaw.data[(i * 784)..<((i + 1) * 784)])
        testImages.append(Matrix(rows: 784, cols: 1, data: imageData))
    }
    
    // One-hot encode labels
    let trainLabels = oneHotEncode(labels: trainLabelsRaw.data.map { Int($0) }, numClasses: 10)
    let testLabels = oneHotEncode(labels: testLabelsRaw.data.map { Int($0) }, numClasses: 10)
    
    return (trainImages, trainLabels, testImages, testLabels)
}
```

### Training on MNIST

```swift
import MLSwift

func trainMNIST() throws {
    print("Loading MNIST dataset...")
    let (trainImages, trainLabels, testImages, testLabels) = try loadMNIST()
    print("Train: \(trainImages.count), Test: \(testImages.count)")
    
    // Create model
    print("Creating model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 784, outputSize: 256))
    model.add(BatchNormLayer(numFeatures: 256))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
    model.add(DenseLayer(inputSize: 256, outputSize: 128))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 128, outputSize: 10))
    model.add(SoftmaxLayer())
    
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // Train
    print("Training...")
    model.train(
        trainInputs: trainImages,
        trainTargets: trainLabels,
        testInputs: testImages,
        testTargets: testLabels,
        epochs: 10,
        batchSize: 64,
        learningRate: 0.01
    )
    
    // Final evaluation
    var correct = 0
    for layer in model.getLayers() {
        if let d = layer as? DropoutLayer { d.training = false }
        if let b = layer as? BatchNormLayer { b.training = false }
    }
    
    for (image, label) in zip(testImages, testLabels) {
        let output = model.forward(image)
        let predicted = output.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        let actual = label.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        if predicted == actual { correct += 1 }
    }
    
    let accuracy = Float(correct) / Float(testImages.count) * 100
    print("Test Accuracy: \(String(format: "%.2f", accuracy))%")
    
    // Save model
    try model.save(to: URL(fileURLWithPath: "mnist_model.json"))
}
```

### Comparison: Python vs Swift

| Feature | Python (NumPy/TensorFlow) | Swift (MLSwift) |
|---------|---------------------------|-----------------|
| Load dataset | `tfds.load()` | `loadBinaryMatrix()` |
| Normalize | `array / 255.0` | `minMaxNormalize()` |
| One-hot | `tf.one_hot()` | `oneHotEncode()` |
| Train/test split | `train_test_split()` | `trainValidationSplit()` |
| Shuffle | `np.random.shuffle()` | `shuffleDataset()` |

## Data Augmentation (Roadmap)

Data augmentation is planned for future releases:

```swift
// Future API (not yet implemented)
/*
func augmentImage(_ image: Matrix, config: AugmentationConfig) -> Matrix {
    var augmented = image
    
    if config.horizontalFlip && Bool.random() {
        augmented = flipHorizontal(augmented)
    }
    
    if config.rotation > 0 {
        let angle = Float.random(in: -config.rotation...config.rotation)
        augmented = rotate(augmented, by: angle)
    }
    
    if config.noise > 0 {
        augmented = addNoise(augmented, scale: config.noise)
    }
    
    return augmented
}
*/
```

## Next Steps

Continue to [Part 7: Roadmap & Future Features](07-roadmap.md) to learn about:
- Planned features (CNNs, RNNs, etc.)
- Contributing guidelines
- Known limitations

---

[← Part 5: Apple Framework Integration](05-apple-integration.md) | [Part 7: Roadmap & Future Features →](07-roadmap.md)
