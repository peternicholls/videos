# Part 3: Building Neural Networks

This part covers how to build and train neural networks using MLSwift's layer abstraction and model architecture.

## Table of Contents

1. [The Layer Protocol](#the-layer-protocol)
2. [Available Layers](#available-layers)
3. [Building a Sequential Model](#building-a-sequential-model)
4. [Training Your Model](#training-your-model)
5. [Evaluating Performance](#evaluating-performance)
6. [Complete Examples](#complete-examples)
7. [Next Steps](#next-steps)

---

## The Layer Protocol

All neural network layers in MLSwift conform to the `Layer` protocol:

```swift
public protocol Layer {
    /// Forward pass: compute output from input
    func forward(_ input: Matrix) -> Matrix
    
    /// Backward pass: compute gradients
    func backward(_ gradOutput: Matrix) -> Matrix
    
    /// Get trainable parameters (weights, biases)
    func parameters() -> [Matrix]
    
    /// Get gradients for parameters
    func gradients() -> [Matrix]
    
    /// Update parameters using learning rate
    func updateParameters(learningRate: Float)
}
```

This protocol enables:
- **Composability**: Stack any layers together
- **Automatic differentiation**: Chain rule applied automatically
- **Extensibility**: Create custom layers easily

## Available Layers

### DenseLayer (Fully Connected)

The core building block—every input connected to every output:

```swift
// Create a dense layer: 784 inputs → 128 outputs
let layer = DenseLayer(inputSize: 784, outputSize: 128)

// Forward pass
let output = layer.forward(input)  // Shape: (128, 1)

// Backward pass
let gradInput = layer.backward(gradOutput)
```

**Parameters**:
- Weights: `(outputSize × inputSize)` matrix
- Biases: `(outputSize × 1)` matrix

### Activation Layers

Wrap activation functions as layers for easy model building:

```swift
// ReLU: max(0, x)
let relu = ReLULayer()

// Softmax: probability distribution
let softmax = SoftmaxLayer()

// Sigmoid: squash to (0, 1)
let sigmoid = SigmoidLayer()

// Tanh: squash to (-1, 1)
let tanh = TanhLayer()
```

**Note**: Activation layers have no trainable parameters.

### DropoutLayer (Regularization)

Randomly zeros neurons during training to prevent overfitting:

```swift
// 30% of neurons set to zero during training
let dropout = DropoutLayer(dropoutRate: 0.3)

// Important: Set to inference mode for predictions
dropout.training = false  // Disable dropout for inference
```

**Behavior**:
- Training: Randomly zeros 30% of activations, scales rest by 1/(1-0.3)
- Inference: Passes all values unchanged

### BatchNormLayer (Normalization)

Normalizes activations for faster, more stable training:

```swift
// Normalize across batch for 128 features
let batchNorm = BatchNormLayer(numFeatures: 128)

// Set to inference mode for predictions
batchNorm.training = false  // Use running statistics
```

**Behavior**:
- Training: Normalizes using batch statistics, updates running mean/variance
- Inference: Normalizes using running statistics (fixed)

## Building a Sequential Model

The `SequentialModel` class stacks layers and handles forward/backward passes:

```swift
import MLSwift

// Create an empty model
let model = SequentialModel()

// Add layers in order
model.add(DenseLayer(inputSize: 784, outputSize: 256))
model.add(BatchNormLayer(numFeatures: 256))
model.add(ReLULayer())
model.add(DropoutLayer(dropoutRate: 0.3))

model.add(DenseLayer(inputSize: 256, outputSize: 128))
model.add(BatchNormLayer(numFeatures: 128))
model.add(ReLULayer())
model.add(DropoutLayer(dropoutRate: 0.2))

model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())

// Set loss function
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
```

### Architecture Patterns

**Pattern 1: Simple Classifier**
```swift
// Input → Dense → ReLU → Dense → Softmax
model.add(DenseLayer(inputSize: features, outputSize: hidden))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: hidden, outputSize: classes))
model.add(SoftmaxLayer())
```

**Pattern 2: Deep Network with Regularization**
```swift
// Each block: Dense → BatchNorm → Activation → Dropout
for (inSize, outSize) in [(784, 512), (512, 256), (256, 128)] {
    model.add(DenseLayer(inputSize: inSize, outputSize: outSize))
    model.add(BatchNormLayer(numFeatures: outSize))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
}
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())
```

**Pattern 3: Binary Classifier**
```swift
model.add(DenseLayer(inputSize: features, outputSize: 64))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 64, outputSize: 1))
model.add(SigmoidLayer())
// Use: Loss.binaryCrossEntropy
```

## Training Your Model

### Single Sample Training

For learning or debugging:

```swift
// Train on one sample at a time
let loss = model.trainStep(input: input, target: target, learningRate: 0.01)
print("Loss: \(loss)")
```

### Batch Training

The `train()` method handles epochs, batching, and progress reporting:

```swift
model.train(
    trainInputs: trainData,      // [Matrix] of inputs
    trainTargets: trainLabels,   // [Matrix] of targets
    testInputs: testData,        // For evaluation
    testTargets: testLabels,
    epochs: 20,                  // Number of passes through data
    batchSize: 32,               // Samples per gradient update
    learningRate: 0.01           // Step size for updates
)
```

### Custom Training Loop

For more control:

```swift
let learningRate: Float = 0.01
let epochs = 50
let batchSize = 32

for epoch in 1...epochs {
    // Shuffle data
    let (shuffledInputs, shuffledTargets) = shuffleDataset(
        inputs: trainInputs, 
        targets: trainTargets
    )
    
    // Create batches
    let batches = createBatches(
        inputs: shuffledInputs, 
        targets: shuffledTargets, 
        batchSize: batchSize
    )
    
    var totalLoss: Float = 0.0
    
    for batch in batches {
        for (input, target) in zip(batch.inputs, batch.targets) {
            totalLoss += model.trainStep(
                input: input, 
                target: target, 
                learningRate: learningRate
            )
        }
    }
    
    let avgLoss = totalLoss / Float(trainInputs.count)
    print("Epoch \(epoch): Loss = \(String(format: "%.4f", avgLoss))")
}
```

## Evaluating Performance

### Making Predictions

```swift
// Single prediction
let output = model.forward(input)

// Get predicted class
let predictedClass = output.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
```

### Computing Accuracy

```swift
func computeAccuracy(model: SequentialModel, inputs: [Matrix], targets: [Matrix]) -> Float {
    var correct = 0
    
    for (input, target) in zip(inputs, targets) {
        let output = model.forward(input)
        
        // Get predicted and true class
        let predicted = output.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        let actual = target.data.enumerated().max(by: { $0.1 < $1.1 })?.offset ?? 0
        
        if predicted == actual {
            correct += 1
        }
    }
    
    return Float(correct) / Float(inputs.count) * 100.0
}

// Usage
let trainAccuracy = computeAccuracy(model: model, inputs: trainInputs, targets: trainTargets)
let testAccuracy = computeAccuracy(model: model, inputs: testInputs, targets: testTargets)

print("Train Accuracy: \(String(format: "%.2f", trainAccuracy))%")
print("Test Accuracy: \(String(format: "%.2f", testAccuracy))%")
```

### Setting Inference Mode

Before making predictions, disable training-specific behaviors:

```swift
// Disable dropout and use batch norm running statistics
for layer in model.getLayers() {
    if let dropout = layer as? DropoutLayer {
        dropout.training = false
    }
    if let batchNorm = layer as? BatchNormLayer {
        batchNorm.training = false
    }
}
```

## Complete Examples

### Example 1: XOR Problem

```swift
import MLSwift

// Network for XOR (2 inputs, 4 hidden, 1 output)
let model = SequentialModel()
model.add(DenseLayer(inputSize: 2, outputSize: 4))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 4, outputSize: 1))
model.setLoss(Loss.meanSquaredError, gradient: Loss.meanSquaredErrorBackward)

// Training data
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
for _ in 1...2000 {
    for (input, target) in zip(inputs, targets) {
        model.trainStep(input: input, target: target, learningRate: 0.5)
    }
}

// Test
for (input, target) in zip(inputs, targets) {
    let output = model.forward(input)
    print("[\(input[0,0]), \(input[1,0])] → \(String(format: "%.3f", output[0,0])) (expected: \(target[0,0]))")
}
```

### Example 2: Multi-class Classification

```swift
import MLSwift

// Generate synthetic dataset (3 classes, 10 features)
var trainInputs: [Matrix] = []
var trainTargets: [Matrix] = []

for _ in 0..<300 {
    let classLabel = Int.random(in: 0..<3)
    
    // Features vary by class
    var features = [Float](repeating: 0, count: 10)
    for i in 0..<10 {
        features[i] = Float.random(in: -1...1) + Float(classLabel) * 0.5
    }
    trainInputs.append(Matrix(rows: 10, cols: 1, data: features))
    
    // One-hot target
    var oneHot = [Float](repeating: 0.0, count: 3)
    oneHot[classLabel] = 1.0
    trainTargets.append(Matrix(rows: 3, cols: 1, data: oneHot))
}

// Build model
let model = SequentialModel()
model.add(DenseLayer(inputSize: 10, outputSize: 32))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 32, outputSize: 16))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 16, outputSize: 3))
model.add(SoftmaxLayer())
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Train
for epoch in 1...100 {
    var totalLoss: Float = 0.0
    for (input, target) in zip(trainInputs, trainTargets) {
        totalLoss += model.trainStep(input: input, target: target, learningRate: 0.01)
    }
    
    if epoch % 20 == 0 {
        let accuracy = computeAccuracy(model: model, inputs: trainInputs, targets: trainTargets)
        print("Epoch \(epoch): Loss = \(String(format: "%.4f", totalLoss / 300)), Accuracy = \(String(format: "%.1f", accuracy))%")
    }
}
```

### Example 3: Network with Regularization

```swift
import MLSwift

let model = SequentialModel()

// Input layer with batch normalization
model.add(DenseLayer(inputSize: 784, outputSize: 256))
model.add(BatchNormLayer(numFeatures: 256))
model.add(ReLULayer())
model.add(DropoutLayer(dropoutRate: 0.5))

// Hidden layer
model.add(DenseLayer(inputSize: 256, outputSize: 128))
model.add(BatchNormLayer(numFeatures: 128))
model.add(TanhLayer())
model.add(DropoutLayer(dropoutRate: 0.3))

// Output layer
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())

model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Train
model.train(
    trainInputs: trainData,
    trainTargets: trainLabels,
    testInputs: testData,
    testTargets: testLabels,
    epochs: 20,
    batchSize: 64,
    learningRate: 0.001
)

// Switch to inference mode
for layer in model.getLayers() {
    if let dropout = layer as? DropoutLayer { dropout.training = false }
    if let batchNorm = layer as? BatchNormLayer { batchNorm.training = false }
}

// Make predictions
let prediction = model.forward(testInput)
```

## Next Steps

Continue to [Part 4: Advanced Features](04-advanced-features.md) to learn about:
- Custom optimizers (Adam, RMSprop, SGD with momentum)
- Model serialization (save/load)
- Hyperparameter tuning

---

[← Part 2: Core Concepts](02-core-concepts.md) | [Part 4: Advanced Features →](04-advanced-features.md)
