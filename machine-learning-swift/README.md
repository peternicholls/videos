# MLSwift

A modern Swift neural network library optimized for Apple Silicon with Metal GPU acceleration.

## Quick Start

```swift
import MLSwift

// Create a neural network
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 128))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())

// Set loss and train
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
model.train(trainInputs: data, trainTargets: labels, epochs: 10, batchSize: 32, learningRate: 0.01)

// Save and load models
try model.save(to: URL(fileURLWithPath: "model.json"))
let loadedModel = try SequentialModel.load(from: URL(fileURLWithPath: "model.json"))
```

## Features

| Category | Features |
|----------|----------|
| **Layers** | Dense, ReLU, Softmax, Sigmoid, Tanh, Dropout, BatchNorm |
| **Loss Functions** | Cross-Entropy, MSE, Binary Cross-Entropy |
| **Optimizers** | SGD, SGD+Momentum, Adam, RMSprop |
| **GPU Acceleration** | Metal compute shaders for matrix ops (up to 18x speedup) |
| **Model I/O** | JSON serialization for save/load |

## Requirements

- macOS 13.0+, Xcode 14.0+, Swift 5.9+
- Apple Silicon Mac (M1/M2/M3+) for GPU acceleration

## Installation

```bash
git clone https://github.com/yourusername/MLSwift.git
cd MLSwift/machine-learning-swift
swift build && swift test && swift run MLSwiftExample
```

## Documentation

📚 **[Full Tutorial Series](docs/00-overview.md)** - Comprehensive 7-part guide:

1. [Getting Started](docs/01-getting-started.md) - Installation and first network
2. [Core Concepts](docs/02-core-concepts.md) - Matrix ops, activations, loss functions  
3. [Building Networks](docs/03-building-networks.md) - Layers, models, training
4. [Advanced Features](docs/04-advanced-features.md) - Optimizers, serialization
5. [Apple Integration](docs/05-apple-integration.md) - Metal, Accelerate, CoreML
6. [Data Processing](docs/06-data-processing.md) - Loading, preprocessing, pipelines
7. [Roadmap](docs/07-roadmap.md) - Future features and contributing

## Performance

| Operation | Matrix Size | CPU | GPU | Speedup |
|-----------|-------------|-----|-----|---------|
| Multiply | 512×512 | 180ms | 20ms | **9x** |
| Multiply | 1024×1024 | 1.45s | 80ms | **18x** |

## Roadmap

**Planned features:**
- [ ] Convolutional layers (CNNs)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Data augmentation
- [ ] Full CoreML export
- [ ] Learning rate schedulers
- [ ] Gradient clipping

## License

MIT License - see LICENSE file for details
