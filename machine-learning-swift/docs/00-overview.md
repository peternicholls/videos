# MLSwift Tutorial Series

A comprehensive guide to building neural networks with MLSwift, a modern Swift library optimized for Apple Silicon with Metal GPU acceleration.

## Tutorial Structure

This tutorial is organized into seven parts, designed to take you from beginner to advanced user:

| Part | Title | Description |
|------|-------|-------------|
| [Part 1](01-getting-started.md) | Getting Started | Installation, requirements, and your first neural network |
| [Part 2](02-core-concepts.md) | Core Concepts | Matrix operations, activations, and loss functions |
| [Part 3](03-building-networks.md) | Building Neural Networks | Layers, models, and training |
| [Part 4](04-advanced-features.md) | Advanced Features | Optimizers, regularization, and model serialization |
| [Part 5](05-apple-integration.md) | Apple Framework Integration | Metal, Accelerate, and CoreML |
| [Part 6](06-data-processing.md) | Data Processing | Loading, preprocessing, and data pipelines |
| [Part 7](07-roadmap.md) | Roadmap & Future Features | Planned features and contributing |

## Quick Start

```swift
import MLSwift

// Create a simple neural network
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 128))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())

// Set loss function and train
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
model.train(trainInputs: data, trainTargets: labels, epochs: 10, batchSize: 32, learningRate: 0.01)
```

## Prerequisites

- macOS 13.0+
- Xcode 14.0+
- Swift 5.9+
- Apple Silicon Mac (M1/M2/M3+) for GPU acceleration

## Navigation

Start with [Part 1: Getting Started](01-getting-started.md) to begin your journey with MLSwift.

---
*MLSwift Tutorial Series - Version 1.0*
