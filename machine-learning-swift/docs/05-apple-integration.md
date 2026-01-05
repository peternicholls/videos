# Part 5: Apple Framework Integration

This part covers how MLSwift integrates with Apple's native frameworks: Metal, Accelerate, and CoreML.

## Table of Contents

1. [Metal GPU Acceleration](#metal-gpu-acceleration)
2. [Accelerate Framework](#accelerate-framework)
3. [CoreML Integration](#coreml-integration)
4. [Performance Optimization](#performance-optimization)
5. [Next Steps](#next-steps)

---

## Metal GPU Acceleration

Metal is Apple's low-level GPU framework. MLSwift uses Metal compute shaders for fast matrix operations.

### How Metal Works in MLSwift

MLSwift automatically uses Metal for large operations:

```
1. Data transferred to GPU buffers
2. Metal compute shader executes
3. Results transferred back to CPU
```

For small operations, CPU is faster (due to transfer overhead).

### Metal Compute Shaders

MLSwift includes 14 Metal compute shaders:

| Shader | Operation | Use Case |
|--------|-----------|----------|
| `matrix_multiply` | A × B | Dense layer forward |
| `matrix_multiply_transA` | A^T × B | Backpropagation |
| `matrix_multiply_transB` | A × B^T | Weight gradients |
| `matrix_add` | A + B | Bias addition |
| `matrix_subtract` | A - B | Gradient computation |
| `matrix_scale` | α × A | Learning rate scaling |
| `relu_forward` | max(0, x) | ReLU activation |
| `relu_backward` | Gradient | ReLU backprop |
| `softmax_*` | exp/sum | Softmax (multi-stage) |
| `cross_entropy_*` | Loss | Cross-entropy loss |

### Example: Matrix Multiplication Shader

```metal
kernel void matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### Using GPU Operations Explicitly

```swift
import MLSwift

// Large matrices benefit from GPU
let a = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)
let b = Matrix(rows: 1024, cols: 1024, randomInRange: -1.0, 1.0)

// Explicit GPU multiplication
do {
    let result = try Matrix.multiplyGPU(a, b)
    print("GPU multiplication completed: \(result.rows)×\(result.cols)")
} catch MetalError.deviceNotFound {
    print("Metal not available, falling back to CPU")
    let result = Matrix.multiply(a, b)
}
```

### Metal Device Management

```swift
import MLSwift

// Check if Metal is available
if let device = MTLCreateSystemDefaultDevice() {
    print("Metal device: \(device.name)")
    print("Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
    print("Unified memory: \(device.hasUnifiedMemory)")
} else {
    print("Metal not available")
}

// MLSwift handles device management automatically via MetalDevice singleton
```

### Thread Group Sizing

MLSwift automatically configures thread groups for optimal performance:

```swift
// For 2D operations (matrix multiply):
// - Thread group size: 16×16 = 256 threads
// - Grid size: ceil(N/16) × ceil(M/16)

// For 1D operations (element-wise):
// - Thread group size: 256 threads
// - Grid size: ceil(elements/256)
```

## Accelerate Framework

Accelerate provides optimized CPU routines for vector and matrix operations.

### vDSP (Vector Digital Signal Processing)

Fast vector operations:

```swift
import Accelerate
import MLSwift

// Example: Normalize data using vDSP
func normalizeWithAccelerate(_ matrix: Matrix) -> (normalized: Matrix, mean: Float, stdDev: Float) {
    var mean: Float = 0.0
    var stdDev: Float = 0.0
    
    // Compute mean
    matrix.data.withUnsafeBufferPointer { ptr in
        vDSP_meanv(ptr.baseAddress!, 1, &mean, vDSP_Length(ptr.count))
    }
    
    // Compute standard deviation
    var subtracted = [Float](repeating: 0.0, count: matrix.data.count)
    var negMean = -mean
    
    matrix.data.withUnsafeBufferPointer { dataPtr in
        subtracted.withUnsafeMutableBufferPointer { subPtr in
            vDSP_vsadd(dataPtr.baseAddress!, 1, &negMean, 
                       subPtr.baseAddress!, 1, vDSP_Length(dataPtr.count))
        }
    }
    
    var sumOfSquares: Float = 0.0
    subtracted.withUnsafeBufferPointer { subPtr in
        vDSP_svesq(subPtr.baseAddress!, 1, &sumOfSquares, vDSP_Length(subPtr.count))
    }
    stdDev = sqrt(sumOfSquares / Float(matrix.data.count))
    
    // Normalize
    guard stdDev > 0 else { return (matrix, mean, 0) }
    
    var normalized = [Float](repeating: 0.0, count: matrix.data.count)
    var invStdDev = 1.0 / stdDev
    
    subtracted.withUnsafeBufferPointer { subPtr in
        normalized.withUnsafeMutableBufferPointer { normPtr in
            vDSP_vsmul(subPtr.baseAddress!, 1, &invStdDev, 
                       normPtr.baseAddress!, 1, vDSP_Length(subPtr.count))
        }
    }
    
    return (Matrix(rows: matrix.rows, cols: matrix.cols, data: normalized), mean, stdDev)
}
```

### BLAS (Basic Linear Algebra Subprograms)

Optimized matrix operations:

```swift
import Accelerate
import MLSwift

// Matrix multiplication using BLAS
func matrixMultiplyBLAS(_ a: Matrix, _ b: Matrix) -> Matrix? {
    guard a.cols == b.rows else { return nil }
    
    var result = [Float](repeating: 0.0, count: a.rows * b.cols)
    
    a.data.withUnsafeBufferPointer { aPtr in
        b.data.withUnsafeBufferPointer { bPtr in
            result.withUnsafeMutableBufferPointer { resultPtr in
                // cblas_sgemm: C = alpha*A*B + beta*C
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    Int32(a.rows), Int32(b.cols), Int32(a.cols),
                    1.0,  // alpha
                    aPtr.baseAddress!, Int32(a.cols),
                    bPtr.baseAddress!, Int32(b.cols),
                    0.0,  // beta
                    resultPtr.baseAddress!, Int32(b.cols)
                )
            }
        }
    }
    
    return Matrix(rows: a.rows, cols: b.cols, data: result)
}
```

### Common vDSP Functions

| Function | Operation | Use Case |
|----------|-----------|----------|
| `vDSP_vadd` | a + b | Vector addition |
| `vDSP_vsub` | a - b | Vector subtraction |
| `vDSP_vmul` | a * b | Element-wise multiply |
| `vDSP_vsmul` | α * a | Scalar multiply |
| `vDSP_meanv` | mean(a) | Mean computation |
| `vDSP_sve` | sum(a) | Sum computation |
| `vDSP_svesq` | sum(a²) | Sum of squares |
| `vDSP_vclip` | clip(a, lo, hi) | Value clipping |

## CoreML Integration

CoreML allows deploying trained models to iOS/macOS apps.

### Export Concept

MLSwift models can be exported to CoreML format (full implementation pending):

```swift
import Foundation
import CoreML
import MLSwift

// Conceptual export (simplified)
func exportModelToCoreML(model: SequentialModel, to url: URL) throws {
    // 1. Extract architecture and weights
    let layers = model.getLayers()
    
    // 2. Build CoreML neural network spec
    // (Requires CoreMLTools or manual spec building)
    
    // 3. Save as .mlmodel
    print("CoreML export: Full implementation pending")
    print("For now, save model using MLSwift's native format")
    try model.save(to: url.deletingPathExtension().appendingPathExtension("json"))
}
```

### Using CoreML Models

Load and use CoreML models for inference:

```swift
import CoreML
import Foundation

func useCoreMLModel() throws {
    // Load compiled model
    let modelURL = URL(fileURLWithPath: "MyModel.mlmodelc")
    let mlModel = try MLModel(contentsOf: modelURL)
    
    // Prepare input
    let inputArray = try MLMultiArray(shape: [784] as [NSNumber], dataType: .float32)
    for i in 0..<784 {
        inputArray[i] = NSNumber(value: Float.random(in: 0...1))
    }
    
    let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
    
    // Predict
    let prediction = try mlModel.prediction(from: input)
    
    if let output = prediction.featureValue(for: "output")?.multiArrayValue {
        print("Prediction: \(output)")
    }
}
```

### Recommended Workflow

1. **Train** with MLSwift (GPU-accelerated training)
2. **Export** model architecture and weights to JSON
3. **Use** CreateML or CoreML APIs for production deployment
4. **Deploy** CoreML model to iOS/macOS apps

```swift
// Export model to JSON (pure Swift)
try model.exportForCoreML(
    to: URL(fileURLWithPath: "model_spec.json"),
    inputShape: [1, 784]
)

// Generate Swift code documenting the architecture
let swiftCode = model.generateCoreMLSwiftCode(inputShape: [1, 784])
print(swiftCode)

// Load the specification back
let spec = try CoreMLExport.loadSpecification(
    from: URL(fileURLWithPath: "model_spec.json")
)
// Access weights from spec["layers"]
```

## Performance Optimization

### When to Use GPU vs CPU

| Matrix Size | Recommended | Reason |
|-------------|-------------|--------|
| < 64×64 | CPU | Transfer overhead dominates |
| 64×64 - 256×256 | Either | Similar performance |
| > 256×256 | GPU | Significant speedup |

### Memory Considerations

```swift
// GPU memory is shared on Apple Silicon (unified memory)
// But be mindful of:
// 1. Large batch sizes consume more memory
// 2. Deep networks need memory for activations
// 3. Gradient storage doubles memory needs

// Tip: Use smaller batches if you run out of memory
let safeBatchSize = 32  // Usually safe
let largeBatchSize = 128  // May need more memory
```

### Profiling Performance

```swift
import Foundation

func profileMatrixMultiply() {
    let sizes = [128, 256, 512, 1024]
    
    for size in sizes {
        let a = Matrix(rows: size, cols: size, randomInRange: -1.0, 1.0)
        let b = Matrix(rows: size, cols: size, randomInRange: -1.0, 1.0)
        
        // CPU timing
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let _ = Matrix.multiply(a, b)
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
        
        // GPU timing (if available)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let _ = try? Matrix.multiplyGPU(a, b)
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
        
        let speedup = cpuTime / gpuTime
        print("\(size)×\(size): CPU=\(String(format: "%.3f", cpuTime))s, GPU=\(String(format: "%.3f", gpuTime))s, Speedup=\(String(format: "%.1f", speedup))x")
    }
}
```

### Optimization Tips

1. **Batch operations**: Process multiple samples together
2. **Preallocate**: Reuse matrices when possible
3. **Avoid copies**: Use views/slices when supported
4. **Profile first**: Measure before optimizing

## Example: Full Pipeline with Accelerate

```swift
import MLSwift
import Accelerate
import Foundation

func optimizedTrainingPipeline() {
    // 1. Load and normalize data with Accelerate
    print("Loading and normalizing data...")
    var rawData = loadRawData()  // Your data loading
    
    // Normalize using vDSP
    var mean: Float = 0.0
    vDSP_meanv(rawData, 1, &mean, vDSP_Length(rawData.count))
    
    var centered = [Float](repeating: 0.0, count: rawData.count)
    var negMean = -mean
    vDSP_vsadd(rawData, 1, &negMean, &centered, 1, vDSP_Length(rawData.count))
    
    // 2. Create model
    print("Creating model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 784, outputSize: 256))
    model.add(ReLULayer())
    model.add(DenseLayer(inputSize: 256, outputSize: 10))
    model.add(SoftmaxLayer())
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // 3. Train with automatic GPU acceleration
    print("Training (GPU-accelerated)...")
    let startTime = CFAbsoluteTimeGetCurrent()
    
    model.train(
        trainInputs: trainInputs,
        trainTargets: trainTargets,
        testInputs: testInputs,
        testTargets: testLabels,
        epochs: 10,
        batchSize: 64,
        learningRate: 0.01
    )
    
    let trainingTime = CFAbsoluteTimeGetCurrent() - startTime
    print("Training completed in \(String(format: "%.2f", trainingTime)) seconds")
    
    // 4. Save for later use
    try? model.save(to: URL(fileURLWithPath: "optimized_model.json"))
}
```

## Next Steps

Continue to [Part 6: Data Processing](06-data-processing.md) to learn about:
- Loading datasets from files
- Data preprocessing and normalization
- Building complete data pipelines

---

[← Part 4: Advanced Features](04-advanced-features.md) | [Part 6: Data Processing →](06-data-processing.md)
