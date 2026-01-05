# Part 2: Core Concepts

This part covers the fundamental building blocks of MLSwift: matrices, activation functions, and loss functions.

## Table of Contents

1. [Matrix Operations](#matrix-operations)
2. [GPU Acceleration with Metal](#gpu-acceleration-with-metal)
3. [Activation Functions](#activation-functions)
4. [Loss Functions](#loss-functions)
5. [Next Steps](#next-steps)

---

## Matrix Operations

Matrices are the foundation of neural networks. MLSwift provides a `Matrix` type optimized for machine learning.

### Creating Matrices

```swift
import MLSwift

// Create a matrix with specific values
let a = Matrix(rows: 2, cols: 3, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

// Create a zero-filled matrix
let zeros = Matrix(rows: 3, cols: 3)

// Create a matrix with random values
let random = Matrix(rows: 100, cols: 100, randomInRange: -1.0, 1.0)
```

### Matrix Storage

Matrices use **row-major order**. Element `(i, j)` is at index `i * cols + j`:

```swift
let mat = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
// Layout: row 0: [1, 2, 3]
//         row 1: [4, 5, 6]

// Access elements
print(mat[0, 0])  // 1.0
print(mat[1, 2])  // 6.0
```

### Basic Operations

```swift
let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
let b = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])

// Element-wise addition
let sum = Matrix.add(a, b)

// Element-wise subtraction
let diff = Matrix.subtract(a, b)

// Scalar multiplication
let scaled = Matrix.scale(a, by: 2.0)

// Matrix multiplication
let product = Matrix.multiply(a, b)

// Transpose
let transposed = Matrix.transpose(a)
```

### Matrix Multiplication

Matrix multiplication is the most computationally expensive operation. MLSwift provides both CPU and GPU implementations:

```swift
// CPU multiplication (automatic)
let result = Matrix.multiply(a, b)

// With transpose options
let result2 = Matrix.multiply(a, b, transposeA: false, transposeB: true)
```

## GPU Acceleration with Metal

MLSwift automatically uses Metal for large matrix operations on Apple Silicon.

### Automatic GPU Usage

The library intelligently chooses between CPU and GPU:

```swift
// Small matrices: CPU is faster (overhead of GPU setup)
let small = Matrix.multiply(smallA, smallB)  // Uses CPU

// Large matrices: GPU is much faster
let large = Matrix.multiply(largeA, largeB)  // Uses GPU automatically
```

### Explicit GPU Operations

For guaranteed GPU execution:

```swift
let a = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)
let b = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)

do {
    let result = try Matrix.multiplyGPU(a, b)
} catch {
    print("GPU operation failed: \(error)")
}
```

### Performance Comparison

Typical speedups on Apple Silicon:

| Operation | Size | CPU Time | GPU Time | Speedup |
|-----------|------|----------|----------|---------|
| Matrix Multiply | 512×512 | 180ms | 20ms | **9x** |
| Matrix Multiply | 1024×1024 | 1.45s | 80ms | **18x** |
| ReLU | 1M elements | 3ms | 1ms | **3x** |

### Metal Compute Shaders

MLSwift includes 14 Metal compute shaders:
- Matrix multiplication (standard, transpose variants)
- Element-wise operations (add, subtract, scale)
- Activation functions (ReLU, Softmax)
- Loss function computations

## Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

### ReLU (Rectified Linear Unit)

The most common activation function:

```swift
// ReLU: max(0, x)
let output = Activations.relu(input)

// Backward pass (gradient)
let gradient = Activations.reluBackward(output, gradOutput: gradOutput)
```

**Properties**:
- Fast to compute
- Helps avoid vanishing gradients
- Can cause "dying ReLU" problem (neurons stuck at 0)

### Softmax

Converts values to probabilities (sums to 1):

```swift
// Softmax: exp(x_i) / sum(exp(x_j))
let probabilities = Activations.softmax(input)
```

**Properties**:
- Output values in range (0, 1)
- Sum of outputs equals 1
- Used for multi-class classification

### Sigmoid

Maps values to range (0, 1):

```swift
// Sigmoid: 1 / (1 + exp(-x))
let output = Activations.sigmoid(input)

// Backward pass
let gradient = Activations.sigmoidBackward(output, gradOutput: gradOutput)
```

**Properties**:
- Smooth gradient
- Can suffer from vanishing gradients
- Good for binary classification outputs

### Tanh

Maps values to range (-1, 1):

```swift
// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
let output = Activations.tanh(input)

// Backward pass
let gradient = Activations.tanhBackward(output, gradOutput: gradOutput)
```

**Properties**:
- Zero-centered output
- Stronger gradients than sigmoid
- Good for hidden layers

### Choosing an Activation Function

| Use Case | Recommended Activation |
|----------|------------------------|
| Hidden layers (default) | ReLU |
| Hidden layers (need negative values) | Tanh |
| Binary classification output | Sigmoid |
| Multi-class classification output | Softmax |
| Regression output | None (linear) |

## Loss Functions

Loss functions measure how wrong predictions are. Training minimizes this value.

### Cross-Entropy Loss

For classification tasks:

```swift
// Cross-entropy: -sum(target * log(prediction))
let loss = Loss.crossEntropy(prediction, target)

// Gradient for backpropagation
let gradient = Loss.crossEntropyBackward(prediction, target)
```

**Use with**: Softmax output for multi-class classification

### Mean Squared Error (MSE)

For regression tasks:

```swift
// MSE: mean((prediction - target)^2)
let loss = Loss.meanSquaredError(prediction, target)

// Gradient
let gradient = Loss.meanSquaredErrorBackward(prediction, target)
```

**Use with**: Linear output for continuous value prediction

### Binary Cross-Entropy

For binary classification:

```swift
// BCE: -[target * log(pred) + (1-target) * log(1-pred)]
let loss = Loss.binaryCrossEntropy(prediction, target)

// Gradient
let gradient = Loss.binaryCrossEntropyBackward(prediction, target)
```

**Use with**: Sigmoid output for binary classification

### Choosing a Loss Function

| Task | Loss Function | Output Activation |
|------|---------------|-------------------|
| Multi-class classification | Cross-Entropy | Softmax |
| Binary classification | Binary Cross-Entropy | Sigmoid |
| Regression | Mean Squared Error | None (linear) |

## Putting It Together

Here's how these concepts work together in a network:

```swift
import MLSwift

// Create input (4 features, batch of 1)
let input = Matrix(rows: 4, cols: 1, data: [0.5, -0.3, 0.8, -0.2])

// Layer 1: Dense transformation
let weights1 = Matrix(rows: 8, cols: 4, randomInRange: -0.5, 0.5)
let hidden = Matrix.multiply(weights1, input)

// Layer 1: ReLU activation
let activated = Activations.relu(hidden)

// Layer 2: Dense transformation
let weights2 = Matrix(rows: 3, cols: 8, randomInRange: -0.5, 0.5)
let logits = Matrix.multiply(weights2, activated)

// Layer 2: Softmax activation (3 classes)
let probabilities = Activations.softmax(logits)

// Compute loss against target
let target = Matrix(rows: 3, cols: 1, data: [0.0, 1.0, 0.0])  // Class 1
let loss = Loss.crossEntropy(probabilities, target)

print("Probabilities: \(probabilities.data)")
print("Loss: \(loss)")
```

## Next Steps

Continue to [Part 3: Building Neural Networks](03-building-networks.md) to learn about:
- Layer abstraction (`DenseLayer`, `ReLULayer`, etc.)
- Sequential models
- Training with backpropagation

---

[← Part 1: Getting Started](01-getting-started.md) | [Part 3: Building Neural Networks →](03-building-networks.md)
