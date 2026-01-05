/// ConvolutionalLayers.swift
/// Convolutional neural network layers for image processing
/// Optimized for Apple Silicon

import Foundation

// MARK: - Padding Mode

/// Padding mode for convolution operations
public enum PaddingMode {
    /// No padding (output is smaller)
    case valid
    /// Pad to keep output same size as input (assuming stride 1)
    case same
    /// Custom padding amount
    case custom(Int)
}

// MARK: - Conv2D Layer

/// 2D Convolutional layer
public class Conv2DLayer: Layer {
    /// Input channels
    public let inputChannels: Int
    
    /// Output channels (number of filters)
    public let outputChannels: Int
    
    /// Kernel size (square kernels)
    public let kernelSize: Int
    
    /// Stride for convolution
    public let stride: Int
    
    /// Padding mode
    public let padding: PaddingMode
    
    /// Kernel weights [outputChannels, inputChannels, kernelSize, kernelSize]
    public var weights: Matrix
    
    /// Bias [outputChannels]
    public var bias: Matrix
    
    /// Gradient w.r.t. weights
    private var weightGrad: Matrix
    
    /// Gradient w.r.t. bias
    private var biasGrad: Matrix
    
    /// Cached input for backward pass
    private var cachedInput: Matrix?
    
    /// Cached input shape (batchSize, inputChannels, height, width as single value: channels * height * width)
    private var inputHeight: Int = 0
    private var inputWidth: Int = 0
    
    /// Initialize Conv2D layer
    /// - Parameters:
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels (filters)
    ///   - kernelSize: Size of square kernel
    ///   - stride: Convolution stride (default 1)
    ///   - padding: Padding mode (default .valid)
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: PaddingMode = .valid
    ) {
        precondition(inputChannels > 0, "Input channels must be positive")
        precondition(outputChannels > 0, "Output channels must be positive")
        precondition(kernelSize > 0, "Kernel size must be positive")
        precondition(stride > 0, "Stride must be positive")
        
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        
        // Kaiming/He initialization for ReLU activation
        let fanIn = inputChannels * kernelSize * kernelSize
        let bound = sqrt(2.0 / Float(fanIn))
        
        // Flatten weights: outputChannels * inputChannels * kernelSize * kernelSize
        let weightCount = outputChannels * inputChannels * kernelSize * kernelSize
        self.weights = Matrix(rows: outputChannels, cols: inputChannels * kernelSize * kernelSize, randomInRange: -bound, bound)
        self.bias = Matrix(rows: outputChannels, cols: 1, value: 0.0)
        
        self.weightGrad = Matrix(rows: outputChannels, cols: inputChannels * kernelSize * kernelSize)
        self.biasGrad = Matrix(rows: outputChannels, cols: 1)
    }
    
    /// Compute padding amount
    private func computePadding(inputSize: Int) -> Int {
        switch padding {
        case .valid:
            return 0
        case .same:
            // For same padding with stride 1: pad = (kernelSize - 1) / 2
            return (kernelSize - 1) / 2
        case .custom(let p):
            return p
        }
    }
    
    /// Compute output size
    private func computeOutputSize(inputSize: Int, pad: Int) -> Int {
        return (inputSize + 2 * pad - kernelSize) / stride + 1
    }
    
    /// Forward pass through the convolutional layer
    /// Input format: (inputChannels * height * width, 1) column vector
    /// Output format: (outputChannels * outHeight * outWidth, 1) column vector
    public func forward(_ input: Matrix) -> Matrix {
        // Infer input dimensions (assuming square input for simplicity)
        let inputSize = input.rows / inputChannels
        let inputDim = Int(sqrt(Double(inputSize)))
        inputHeight = inputDim
        inputWidth = inputDim
        
        precondition(inputDim * inputDim * inputChannels == input.rows, 
                    "Input must be a flattened image with inputChannels * height * width elements")
        
        cachedInput = input
        
        let pad = computePadding(inputSize: inputDim)
        let outHeight = computeOutputSize(inputSize: inputDim, pad: pad)
        let outWidth = computeOutputSize(inputSize: inputDim, pad: pad)
        
        var output = Matrix(rows: outputChannels * outHeight * outWidth, cols: 1)
        
        // Perform convolution
        for oc in 0..<outputChannels {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    var sum: Float = 0.0
                    
                    // Compute convolution at this position
                    for ic in 0..<inputChannels {
                        for kh in 0..<kernelSize {
                            for kw in 0..<kernelSize {
                                let ih = oh * stride + kh - pad
                                let iw = ow * stride + kw - pad
                                
                                // Check bounds (zero-padding for out of bounds)
                                if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
                                    let inputIdx = ic * inputHeight * inputWidth + ih * inputWidth + iw
                                    let weightIdx = ic * kernelSize * kernelSize + kh * kernelSize + kw
                                    sum += input.data[inputIdx] * weights[oc, weightIdx]
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    sum += bias.data[oc]
                    
                    let outputIdx = oc * outHeight * outWidth + oh * outWidth + ow
                    output.data[outputIdx] = sum
                }
            }
        }
        
        return output
    }
    
    /// Backward pass through the convolutional layer
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let input = cachedInput else {
            fatalError("Forward must be called before backward")
        }
        
        let pad = computePadding(inputSize: inputHeight)
        let outHeight = computeOutputSize(inputSize: inputHeight, pad: pad)
        let outWidth = computeOutputSize(inputSize: inputWidth, pad: pad)
        
        // Zero gradients
        weightGrad.zero()
        biasGrad.zero()
        
        var gradInput = Matrix(rows: inputChannels * inputHeight * inputWidth, cols: 1)
        
        // Compute gradients
        for oc in 0..<outputChannels {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    let outputIdx = oc * outHeight * outWidth + oh * outWidth + ow
                    let gradOut = gradOutput.data[outputIdx]
                    
                    // Gradient w.r.t. bias
                    biasGrad.data[oc] += gradOut
                    
                    // Gradient w.r.t. weights and input
                    for ic in 0..<inputChannels {
                        for kh in 0..<kernelSize {
                            for kw in 0..<kernelSize {
                                let ih = oh * stride + kh - pad
                                let iw = ow * stride + kw - pad
                                
                                if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
                                    let inputIdx = ic * inputHeight * inputWidth + ih * inputWidth + iw
                                    let weightIdx = ic * kernelSize * kernelSize + kh * kernelSize + kw
                                    
                                    // Gradient w.r.t. weights: dW += gradOutput * input
                                    weightGrad[oc, weightIdx] += gradOut * input.data[inputIdx]
                                    
                                    // Gradient w.r.t. input: dInput += gradOutput * weights
                                    gradInput.data[inputIdx] += gradOut * weights[oc, weightIdx]
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return gradInput
    }
    
    public func parameters() -> [Matrix] {
        return [weights, bias]
    }
    
    public func gradients() -> [Matrix] {
        return [weightGrad, biasGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        var scaledWeightGrad = weightGrad
        scaledWeightGrad.scale(by: learningRate)
        
        var scaledBiasGrad = biasGrad
        scaledBiasGrad.scale(by: learningRate)
        
        weights = Matrix.subtract(weights, scaledWeightGrad)
        bias = Matrix.subtract(bias, scaledBiasGrad)
        
        weightGrad.zero()
        biasGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        weightGrad.scale(by: scale)
        biasGrad.scale(by: scale)
    }
}

// MARK: - MaxPool2D Layer

/// 2D Max Pooling layer
public class MaxPool2DLayer: Layer {
    /// Pool size (square pooling)
    public let poolSize: Int
    
    /// Stride (defaults to poolSize for non-overlapping pooling)
    public let stride: Int
    
    /// Cached max indices for backward pass
    private var cachedMaxIndices: [Int] = []
    
    /// Cached input shape
    private var inputChannels: Int = 0
    private var inputHeight: Int = 0
    private var inputWidth: Int = 0
    
    /// Initialize MaxPool2D layer
    /// - Parameters:
    ///   - poolSize: Size of pooling window
    ///   - stride: Stride (default is poolSize for non-overlapping)
    public init(poolSize: Int, stride: Int? = nil) {
        precondition(poolSize > 0, "Pool size must be positive")
        self.poolSize = poolSize
        self.stride = stride ?? poolSize
    }
    
    public func forward(_ input: Matrix) -> Matrix {
        // Infer input dimensions (channels, height, width)
        // Assume square spatial dimensions
        let totalElements = input.rows
        
        // Try to infer channels - common values are 1, 3, 32, 64, etc.
        // For simplicity, we assume the caller provides correct dimensions
        // We'll try common channel counts
        var foundChannels = 0
        for channels in [1, 3, 16, 32, 64, 128, 256, 512] {
            let spatialSize = totalElements / channels
            let dim = Int(sqrt(Double(spatialSize)))
            if dim * dim * channels == totalElements {
                foundChannels = channels
                inputChannels = channels
                inputHeight = dim
                inputWidth = dim
                break
            }
        }
        
        if foundChannels == 0 {
            // Fallback: assume single channel
            inputChannels = 1
            let spatialSize = totalElements
            let dim = Int(sqrt(Double(spatialSize)))
            inputHeight = dim
            inputWidth = dim
        }
        
        let outHeight = (inputHeight - poolSize) / stride + 1
        let outWidth = (inputWidth - poolSize) / stride + 1
        
        var output = Matrix(rows: inputChannels * outHeight * outWidth, cols: 1)
        cachedMaxIndices = [Int](repeating: 0, count: output.rows)
        
        for c in 0..<inputChannels {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    var maxVal: Float = -.infinity
                    var maxIdx = 0
                    
                    for ph in 0..<poolSize {
                        for pw in 0..<poolSize {
                            let ih = oh * stride + ph
                            let iw = ow * stride + pw
                            let inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw
                            
                            if input.data[inputIdx] > maxVal {
                                maxVal = input.data[inputIdx]
                                maxIdx = inputIdx
                            }
                        }
                    }
                    
                    let outputIdx = c * outHeight * outWidth + oh * outWidth + ow
                    output.data[outputIdx] = maxVal
                    cachedMaxIndices[outputIdx] = maxIdx
                }
            }
        }
        
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        var gradInput = Matrix(rows: inputChannels * inputHeight * inputWidth, cols: 1)
        
        // Route gradients to max positions
        for i in 0..<gradOutput.rows {
            let maxIdx = cachedMaxIndices[i]
            gradInput.data[maxIdx] += gradOutput.data[i]
        }
        
        return gradInput
    }
    
    public func parameters() -> [Matrix] { return [] }
    public func gradients() -> [Matrix] { return [] }
    public func updateParameters(learningRate: Float) {}
    public func scaleGradients(by scale: Float) {}
}

// MARK: - Average Pooling Layer

/// 2D Average Pooling layer
public class AvgPool2DLayer: Layer {
    /// Pool size (square pooling)
    public let poolSize: Int
    
    /// Stride
    public let stride: Int
    
    /// Cached input shape
    private var inputChannels: Int = 0
    private var inputHeight: Int = 0
    private var inputWidth: Int = 0
    
    public init(poolSize: Int, stride: Int? = nil) {
        precondition(poolSize > 0, "Pool size must be positive")
        self.poolSize = poolSize
        self.stride = stride ?? poolSize
    }
    
    public func forward(_ input: Matrix) -> Matrix {
        let totalElements = input.rows
        
        // Infer dimensions
        var foundChannels = 0
        for channels in [1, 3, 16, 32, 64, 128, 256, 512] {
            let spatialSize = totalElements / channels
            let dim = Int(sqrt(Double(spatialSize)))
            if dim * dim * channels == totalElements {
                foundChannels = channels
                inputChannels = channels
                inputHeight = dim
                inputWidth = dim
                break
            }
        }
        
        if foundChannels == 0 {
            inputChannels = 1
            let dim = Int(sqrt(Double(totalElements)))
            inputHeight = dim
            inputWidth = dim
        }
        
        let outHeight = (inputHeight - poolSize) / stride + 1
        let outWidth = (inputWidth - poolSize) / stride + 1
        
        var output = Matrix(rows: inputChannels * outHeight * outWidth, cols: 1)
        
        for c in 0..<inputChannels {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    var sum: Float = 0
                    
                    for ph in 0..<poolSize {
                        for pw in 0..<poolSize {
                            let ih = oh * stride + ph
                            let iw = ow * stride + pw
                            let inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw
                            sum += input.data[inputIdx]
                        }
                    }
                    
                    let outputIdx = c * outHeight * outWidth + oh * outWidth + ow
                    output.data[outputIdx] = sum / Float(poolSize * poolSize)
                }
            }
        }
        
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        let outHeight = (inputHeight - poolSize) / stride + 1
        let outWidth = (inputWidth - poolSize) / stride + 1
        
        var gradInput = Matrix(rows: inputChannels * inputHeight * inputWidth, cols: 1)
        let scale = 1.0 / Float(poolSize * poolSize)
        
        for c in 0..<inputChannels {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    let outputIdx = c * outHeight * outWidth + oh * outWidth + ow
                    let grad = gradOutput.data[outputIdx] * scale
                    
                    for ph in 0..<poolSize {
                        for pw in 0..<poolSize {
                            let ih = oh * stride + ph
                            let iw = ow * stride + pw
                            let inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw
                            gradInput.data[inputIdx] += grad
                        }
                    }
                }
            }
        }
        
        return gradInput
    }
    
    public func parameters() -> [Matrix] { return [] }
    public func gradients() -> [Matrix] { return [] }
    public func updateParameters(learningRate: Float) {}
    public func scaleGradients(by scale: Float) {}
}

// MARK: - Flatten Layer

/// Flatten layer - converts multi-dimensional input to 1D
/// Note: In MLSwift, inputs are already column vectors, so this is mostly a no-op
/// but useful for clarity in model architecture
public class FlattenLayer: Layer {
    private var cachedInputShape: (rows: Int, cols: Int)?
    
    public init() {}
    
    public func forward(_ input: Matrix) -> Matrix {
        cachedInputShape = (input.rows, input.cols)
        
        // If already a column vector, return as-is
        if input.cols == 1 {
            return input
        }
        
        // Otherwise, flatten to column vector
        return Matrix(rows: input.rows * input.cols, cols: 1, data: input.data)
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        guard let shape = cachedInputShape else {
            fatalError("Forward must be called before backward")
        }
        
        // Reshape gradient back to original shape
        return Matrix(rows: shape.rows, cols: shape.cols, data: gradOutput.data)
    }
    
    public func parameters() -> [Matrix] { return [] }
    public func gradients() -> [Matrix] { return [] }
    public func updateParameters(learningRate: Float) {}
    public func scaleGradients(by scale: Float) {}
}
