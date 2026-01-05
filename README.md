# videos

Code from videos on my YouTube channel.

## Projects

### Machine Learning Swift (MLSwift)

A modern Swift neural network library optimized for Apple Silicon with Metal GPU acceleration.

📚 **[Tutorial Series](machine-learning-swift/docs/00-overview.md)** - Comprehensive 7-part guide

**Features:**
- Metal GPU acceleration (up to 18x speedup)
- Dense, ReLU, Softmax, Sigmoid, Tanh, Dropout, BatchNorm layers
- Adam, RMSprop, SGD optimizers
- Model save/load (JSON serialization)

**Quick Start:**
```bash
cd machine-learning-swift
swift build && swift test && swift run MLSwiftExample
```

[View Full Documentation →](machine-learning-swift/README.md)

### Machine Learning (C)

Original C implementation of the neural network library.

```bash
cd machine-learning
gcc -o ml main.c -lm
./ml
```

## Documentation

- [Conversion Notes](CONVERSION_NOTES.md) - C to Swift conversion details
- [Refactoring Summary](REFACTORING_SUMMARY.md) - Project overview and status
