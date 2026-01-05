# Part 1: Getting Started

Welcome to MLSwift! This tutorial will guide you through installing MLSwift and building your first neural network.

## Table of Contents

1. [What is MLSwift?](#what-is-mlswift)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Your First Neural Network](#your-first-neural-network)
5. [Understanding the Output](#understanding-the-output)
6. [Next Steps](#next-steps)

---

## What is MLSwift?

MLSwift is a from-scratch neural network library written in Swift, optimized for Apple Silicon. Key features include:

- **Metal GPU Acceleration**: All heavy computations run on Apple's GPU
- **Modern Swift**: Leverages Swift's type safety, protocols, and ARC
- **Apple Silicon Optimized**: Designed specifically for M1/M2/M3+ processors
- **Educational**: Clear, readable code for learning neural network internals

## Requirements

| Requirement | Minimum Version |
|-------------|-----------------|
| macOS | 13.0+ |
| Xcode | 14.0+ |
| Swift | 5.9+ |
| Hardware | Apple Silicon Mac (M1/M2/M3+) |

> **Note**: Intel Macs will work but without Metal GPU acceleration.

## Installation

### Using Swift Package Manager

Add MLSwift to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/MLSwift.git", from: "1.0.0")
]
```

### Local Development

Clone and build the repository:

```bash
git clone https://github.com/yourusername/MLSwift.git
cd MLSwift/machine-learning-swift
swift build
```

### Verify Installation

Run the example program:

```bash
swift run MLSwiftExample
```

You should see output demonstrating XOR learning and multi-class classification.

## Your First Neural Network

Let's solve the classic XOR problem—a simple but fundamental neural network task.

### The XOR Problem

XOR (exclusive or) outputs `1` when inputs differ, `0` when they're the same:

| Input A | Input B | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Complete Code

```swift
import MLSwift

// 1. Create a neural network with 2 inputs, hidden layer of 4, and 1 output
let model = SequentialModel()
model.add(DenseLayer(inputSize: 2, outputSize: 4))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 4, outputSize: 1))

// 2. Set the loss function (Mean Squared Error for regression)
model.setLoss(Loss.meanSquaredError, gradient: Loss.meanSquaredErrorBackward)

// 3. Prepare training data
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

// 4. Train for 1000 epochs
for epoch in 1...1000 {
    var totalLoss: Float = 0.0
    for (input, target) in zip(inputs, targets) {
        totalLoss += model.trainStep(input: input, target: target, learningRate: 0.1)
    }
    
    if epoch % 200 == 0 {
        print("Epoch \(epoch): Loss = \(totalLoss / 4.0)")
    }
}

// 5. Test the trained network
print("\nResults:")
for (input, target) in zip(inputs, targets) {
    let output = model.forward(input)
    let predicted = output[0, 0]
    let expected = target[0, 0]
    print("Input: [\(input[0, 0]), \(input[1, 0])] -> Predicted: \(String(format: "%.3f", predicted)), Expected: \(expected)")
}
```

### Code Breakdown

1. **Create the model**: `SequentialModel` holds layers executed in sequence
2. **Add layers**: 
   - `DenseLayer(2→4)`: Fully connected layer from 2 to 4 neurons
   - `ReLULayer()`: Applies ReLU activation (max(0, x))
   - `DenseLayer(4→1)`: Output layer with 1 neuron
3. **Set loss function**: MSE measures how far predictions are from targets
4. **Train**: `trainStep` performs forward pass, computes loss, backpropagates, and updates weights
5. **Test**: Use `forward()` to make predictions

## Understanding the Output

After training, you should see output similar to:

```
Epoch 200: Loss = 0.234
Epoch 400: Loss = 0.089
Epoch 600: Loss = 0.023
Epoch 800: Loss = 0.008
Epoch 1000: Loss = 0.003

Results:
Input: [0.0, 0.0] -> Predicted: 0.021, Expected: 0.0
Input: [0.0, 1.0] -> Predicted: 0.978, Expected: 1.0
Input: [1.0, 0.0] -> Predicted: 0.981, Expected: 1.0
Input: [1.0, 1.0] -> Predicted: 0.019, Expected: 0.0
```

The network learned to approximate XOR! Values near 0 and 1 indicate successful learning.

## Running Tests

MLSwift includes 21 unit tests:

```bash
swift test
```

All tests should pass, verifying matrix operations, activations, loss functions, and training.

## Next Steps

Continue to [Part 2: Core Concepts](02-core-concepts.md) to learn about:
- Matrix operations and GPU acceleration
- Activation functions (ReLU, Softmax, Sigmoid, Tanh)
- Loss functions and their gradients

---

[← Overview](00-overview.md) | [Part 2: Core Concepts →](02-core-concepts.md)
