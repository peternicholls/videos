/// RecurrentLayers.swift
/// Recurrent neural network layers (LSTM, GRU) for sequence processing
/// Optimized for Apple Silicon

import Foundation

// MARK: - LSTM Layer

/// Long Short-Term Memory (LSTM) layer for sequence processing
public class LSTMLayer: Layer {
    /// Input dimension
    public let inputSize: Int
    
    /// Hidden state dimension
    public let hiddenSize: Int
    
    /// Whether to return sequences (all hidden states) or just final state
    public let returnSequences: Bool
    
    // LSTM has 4 gates: input (i), forget (f), cell (g), output (o)
    // Combined weight matrices for efficiency
    
    /// Input-to-hidden weights [4 * hiddenSize, inputSize]
    public var weightsIH: Matrix
    
    /// Hidden-to-hidden weights [4 * hiddenSize, hiddenSize]
    public var weightsHH: Matrix
    
    /// Biases [4 * hiddenSize, 1]
    public var bias: Matrix
    
    // Gradients
    private var weightsIHGrad: Matrix
    private var weightsHHGrad: Matrix
    private var biasGrad: Matrix
    
    // Cached values for backward pass
    private var cachedInputs: [Matrix] = []
    private var cachedHiddenStates: [Matrix] = []
    private var cachedCellStates: [Matrix] = []
    private var cachedGates: [(i: Matrix, f: Matrix, g: Matrix, o: Matrix)] = []
    
    /// Initialize LSTM layer
    /// - Parameters:
    ///   - inputSize: Dimension of input features
    ///   - hiddenSize: Dimension of hidden state
    ///   - returnSequences: If true, return all hidden states; if false, return only final state
    public init(inputSize: Int, hiddenSize: Int, returnSequences: Bool = false) {
        precondition(inputSize > 0, "Input size must be positive")
        precondition(hiddenSize > 0, "Hidden size must be positive")
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.returnSequences = returnSequences
        
        // Xavier initialization
        let boundIH = sqrt(6.0 / Float(inputSize + hiddenSize))
        let boundHH = sqrt(6.0 / Float(hiddenSize + hiddenSize))
        
        self.weightsIH = Matrix(rows: 4 * hiddenSize, cols: inputSize, randomInRange: -boundIH, boundIH)
        self.weightsHH = Matrix(rows: 4 * hiddenSize, cols: hiddenSize, randomInRange: -boundHH, boundHH)
        self.bias = Matrix(rows: 4 * hiddenSize, cols: 1, value: 0.0)
        
        // Initialize forget gate bias to 1.0 for better gradient flow
        for i in hiddenSize..<(2 * hiddenSize) {
            self.bias.data[i] = 1.0
        }
        
        self.weightsIHGrad = Matrix(rows: 4 * hiddenSize, cols: inputSize)
        self.weightsHHGrad = Matrix(rows: 4 * hiddenSize, cols: hiddenSize)
        self.biasGrad = Matrix(rows: 4 * hiddenSize, cols: 1)
    }
    
    /// Forward pass through LSTM
    /// Input format: (seqLen * inputSize, 1) - concatenated sequence
    /// Output format: (seqLen * hiddenSize, 1) if returnSequences, else (hiddenSize, 1)
    public func forward(_ input: Matrix) -> Matrix {
        // Infer sequence length
        let seqLen = input.rows / inputSize
        precondition(seqLen * inputSize == input.rows, "Input size must be divisible by inputSize")
        
        // Clear cached values
        cachedInputs = []
        cachedHiddenStates = []
        cachedCellStates = []
        cachedGates = []
        
        // Initialize hidden and cell states
        var h = Matrix(rows: hiddenSize, cols: 1)
        var c = Matrix(rows: hiddenSize, cols: 1)
        
        var outputs: [Matrix] = []
        
        // Process sequence
        for t in 0..<seqLen {
            // Extract input at timestep t
            let startIdx = t * inputSize
            let x = Matrix(rows: inputSize, cols: 1, 
                          data: Array(input.data[startIdx..<(startIdx + inputSize)]))
            cachedInputs.append(x)
            
            // Compute gates: [i, f, g, o] = W_ih * x + W_hh * h + b
            let gates = computeGates(x: x, h: h)
            
            // Apply activations
            let i = sigmoid(extractGate(gates, index: 0))  // Input gate
            let f = sigmoid(extractGate(gates, index: 1))  // Forget gate
            let g = tanh(extractGate(gates, index: 2))     // Cell candidate
            let o = sigmoid(extractGate(gates, index: 3))  // Output gate
            
            cachedGates.append((i, f, g, o))
            
            // Update cell state: c = f * c_prev + i * g
            var newC = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                newC.data[j] = f.data[j] * c.data[j] + i.data[j] * g.data[j]
            }
            
            cachedCellStates.append(c)
            c = newC
            
            // Update hidden state: h = o * tanh(c)
            let tanhC = tanh(c)
            var newH = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                newH.data[j] = o.data[j] * tanhC.data[j]
            }
            
            cachedHiddenStates.append(h)
            h = newH
            
            outputs.append(h)
        }
        
        // Store final states for backward pass
        cachedHiddenStates.append(h)
        cachedCellStates.append(c)
        
        if returnSequences {
            // Concatenate all hidden states
            var result = Matrix(rows: seqLen * hiddenSize, cols: 1)
            for t in 0..<seqLen {
                for j in 0..<hiddenSize {
                    result.data[t * hiddenSize + j] = outputs[t].data[j]
                }
            }
            return result
        } else {
            // Return only final hidden state
            return h
        }
    }
    
    /// Backward pass through LSTM
    public func backward(_ gradOutput: Matrix) -> Matrix {
        let seqLen = cachedInputs.count
        
        // Zero gradients
        weightsIHGrad.zero()
        weightsHHGrad.zero()
        biasGrad.zero()
        
        var gradInput = Matrix(rows: seqLen * inputSize, cols: 1)
        
        // Initialize gradients for hidden and cell states
        var dh_next = Matrix(rows: hiddenSize, cols: 1)
        var dc_next = Matrix(rows: hiddenSize, cols: 1)
        
        // If returnSequences, gradOutput is (seqLen * hiddenSize, 1)
        // Otherwise, gradOutput is (hiddenSize, 1)
        
        // Backward through time
        for t in stride(from: seqLen - 1, through: 0, by: -1) {
            // Get gradient at this timestep
            var dh: Matrix
            if returnSequences {
                let startIdx = t * hiddenSize
                dh = Matrix(rows: hiddenSize, cols: 1,
                           data: Array(gradOutput.data[startIdx..<(startIdx + hiddenSize)]))
                dh = Matrix.add(dh, dh_next)
            } else {
                if t == seqLen - 1 {
                    dh = gradOutput
                } else {
                    dh = dh_next
                }
            }
            
            let (i, f, g, o) = cachedGates[t]
            let c_prev = cachedCellStates[t]
            let c = t < seqLen - 1 ? cachedCellStates[t + 1] : cachedCellStates[seqLen]
            let h_prev = cachedHiddenStates[t]
            let x = cachedInputs[t]
            
            // Gradient through h = o * tanh(c)
            let tanhC = tanh(c)
            
            // do = dh * tanh(c)
            var do_gate = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                do_gate.data[j] = dh.data[j] * tanhC.data[j]
            }
            
            // dc = dh * o * (1 - tanh(c)^2) + dc_next
            var dc = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                dc.data[j] = dh.data[j] * o.data[j] * (1 - tanhC.data[j] * tanhC.data[j]) + dc_next.data[j]
            }
            
            // di = dc * g
            var di = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                di.data[j] = dc.data[j] * g.data[j]
            }
            
            // df = dc * c_prev
            var df = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                df.data[j] = dc.data[j] * c_prev.data[j]
            }
            
            // dg = dc * i
            var dg = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                dg.data[j] = dc.data[j] * i.data[j]
            }
            
            // Apply activation gradients
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            // tanh'(x) = 1 - tanh(x)^2
            for j in 0..<hiddenSize {
                di.data[j] *= i.data[j] * (1 - i.data[j])
                df.data[j] *= f.data[j] * (1 - f.data[j])
                dg.data[j] *= (1 - g.data[j] * g.data[j])
                do_gate.data[j] *= o.data[j] * (1 - o.data[j])
            }
            
            // Concatenate gate gradients
            var dGates = Matrix(rows: 4 * hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                dGates.data[j] = di.data[j]
                dGates.data[j + hiddenSize] = df.data[j]
                dGates.data[j + 2 * hiddenSize] = dg.data[j]
                dGates.data[j + 3 * hiddenSize] = do_gate.data[j]
            }
            
            // Accumulate gradients
            // dW_ih += dGates * x^T
            let dWih = Matrix.multiply(dGates, x, transposeB: true)
            weightsIHGrad = Matrix.add(weightsIHGrad, dWih)
            
            // dW_hh += dGates * h_prev^T
            let dWhh = Matrix.multiply(dGates, h_prev, transposeB: true)
            weightsHHGrad = Matrix.add(weightsHHGrad, dWhh)
            
            // db += dGates
            biasGrad = Matrix.add(biasGrad, dGates)
            
            // Gradient w.r.t. input: dx = W_ih^T * dGates
            let dx = Matrix.multiply(weightsIH, dGates, transposeA: true)
            for j in 0..<inputSize {
                gradInput.data[t * inputSize + j] = dx.data[j]
            }
            
            // Gradient w.r.t. previous hidden state: dh_prev = W_hh^T * dGates
            dh_next = Matrix.multiply(weightsHH, dGates, transposeA: true)
            
            // Gradient w.r.t. previous cell state: dc_prev = dc * f
            for j in 0..<hiddenSize {
                dc_next.data[j] = dc.data[j] * f.data[j]
            }
        }
        
        return gradInput
    }
    
    // MARK: - Helper Functions
    
    private func computeGates(x: Matrix, h: Matrix) -> Matrix {
        let xPart = Matrix.multiply(weightsIH, x)
        let hPart = Matrix.multiply(weightsHH, h)
        var gates = Matrix.add(xPart, hPart)
        gates = Matrix.add(gates, bias)
        return gates
    }
    
    private func extractGate(_ gates: Matrix, index: Int) -> Matrix {
        let startIdx = index * hiddenSize
        return Matrix(rows: hiddenSize, cols: 1,
                     data: Array(gates.data[startIdx..<(startIdx + hiddenSize)]))
    }
    
    private func sigmoid(_ x: Matrix) -> Matrix {
        var result = x
        for i in 0..<result.data.count {
            result.data[i] = 1.0 / (1.0 + exp(-result.data[i]))
        }
        return result
    }
    
    private func tanh(_ x: Matrix) -> Matrix {
        var result = x
        for i in 0..<result.data.count {
            result.data[i] = Foundation.tanh(result.data[i])
        }
        return result
    }
    
    public func parameters() -> [Matrix] {
        return [weightsIH, weightsHH, bias]
    }
    
    public func gradients() -> [Matrix] {
        return [weightsIHGrad, weightsHHGrad, biasGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        var scaledIHGrad = weightsIHGrad
        scaledIHGrad.scale(by: learningRate)
        
        var scaledHHGrad = weightsHHGrad
        scaledHHGrad.scale(by: learningRate)
        
        var scaledBiasGrad = biasGrad
        scaledBiasGrad.scale(by: learningRate)
        
        weightsIH = Matrix.subtract(weightsIH, scaledIHGrad)
        weightsHH = Matrix.subtract(weightsHH, scaledHHGrad)
        bias = Matrix.subtract(bias, scaledBiasGrad)
        
        weightsIHGrad.zero()
        weightsHHGrad.zero()
        biasGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        weightsIHGrad.scale(by: scale)
        weightsHHGrad.scale(by: scale)
        biasGrad.scale(by: scale)
    }
}

// MARK: - GRU Layer

/// Gated Recurrent Unit (GRU) layer for sequence processing
/// GRU has fewer parameters than LSTM and often performs similarly
public class GRULayer: Layer {
    /// Input dimension
    public let inputSize: Int
    
    /// Hidden state dimension
    public let hiddenSize: Int
    
    /// Whether to return sequences
    public let returnSequences: Bool
    
    // GRU has 3 gates: reset (r), update (z), and new (n)
    
    /// Input-to-hidden weights [3 * hiddenSize, inputSize]
    public var weightsIH: Matrix
    
    /// Hidden-to-hidden weights [3 * hiddenSize, hiddenSize]
    public var weightsHH: Matrix
    
    /// Biases [3 * hiddenSize, 1]
    public var bias: Matrix
    
    // Gradients
    private var weightsIHGrad: Matrix
    private var weightsHHGrad: Matrix
    private var biasGrad: Matrix
    
    // Cached values for backward pass
    private var cachedInputs: [Matrix] = []
    private var cachedHiddenStates: [Matrix] = []
    private var cachedGates: [(r: Matrix, z: Matrix, n: Matrix)] = []
    
    /// Initialize GRU layer
    public init(inputSize: Int, hiddenSize: Int, returnSequences: Bool = false) {
        precondition(inputSize > 0, "Input size must be positive")
        precondition(hiddenSize > 0, "Hidden size must be positive")
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.returnSequences = returnSequences
        
        let boundIH = sqrt(6.0 / Float(inputSize + hiddenSize))
        let boundHH = sqrt(6.0 / Float(hiddenSize + hiddenSize))
        
        self.weightsIH = Matrix(rows: 3 * hiddenSize, cols: inputSize, randomInRange: -boundIH, boundIH)
        self.weightsHH = Matrix(rows: 3 * hiddenSize, cols: hiddenSize, randomInRange: -boundHH, boundHH)
        self.bias = Matrix(rows: 3 * hiddenSize, cols: 1, value: 0.0)
        
        self.weightsIHGrad = Matrix(rows: 3 * hiddenSize, cols: inputSize)
        self.weightsHHGrad = Matrix(rows: 3 * hiddenSize, cols: hiddenSize)
        self.biasGrad = Matrix(rows: 3 * hiddenSize, cols: 1)
    }
    
    public func forward(_ input: Matrix) -> Matrix {
        let seqLen = input.rows / inputSize
        precondition(seqLen * inputSize == input.rows, "Input size must be divisible by inputSize")
        
        cachedInputs = []
        cachedHiddenStates = []
        cachedGates = []
        
        var h = Matrix(rows: hiddenSize, cols: 1)
        var outputs: [Matrix] = []
        
        for t in 0..<seqLen {
            let startIdx = t * inputSize
            let x = Matrix(rows: inputSize, cols: 1,
                          data: Array(input.data[startIdx..<(startIdx + inputSize)]))
            cachedInputs.append(x)
            cachedHiddenStates.append(h)
            
            // Compute gates
            let xGates = Matrix.multiply(weightsIH, x)
            let hGates = Matrix.multiply(weightsHH, h)
            
            // Extract reset and update gates (computed together)
            var r = Matrix(rows: hiddenSize, cols: 1)
            var z = Matrix(rows: hiddenSize, cols: 1)
            
            for j in 0..<hiddenSize {
                r.data[j] = sigmoid(xGates.data[j] + hGates.data[j] + bias.data[j])
                z.data[j] = sigmoid(xGates.data[j + hiddenSize] + hGates.data[j + hiddenSize] + bias.data[j + hiddenSize])
            }
            
            // Compute new gate: n = tanh(x_n + r * h_n)
            var n = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                let xn = xGates.data[j + 2 * hiddenSize] + bias.data[j + 2 * hiddenSize]
                let hn = hGates.data[j + 2 * hiddenSize]
                n.data[j] = Foundation.tanh(xn + r.data[j] * hn)
            }
            
            cachedGates.append((r, z, n))
            
            // Compute new hidden state: h = (1 - z) * n + z * h_prev
            var newH = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                newH.data[j] = (1 - z.data[j]) * n.data[j] + z.data[j] * h.data[j]
            }
            
            h = newH
            outputs.append(h)
        }
        
        cachedHiddenStates.append(h)
        
        if returnSequences {
            var result = Matrix(rows: seqLen * hiddenSize, cols: 1)
            for t in 0..<seqLen {
                for j in 0..<hiddenSize {
                    result.data[t * hiddenSize + j] = outputs[t].data[j]
                }
            }
            return result
        } else {
            return h
        }
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        let seqLen = cachedInputs.count
        
        weightsIHGrad.zero()
        weightsHHGrad.zero()
        biasGrad.zero()
        
        var gradInput = Matrix(rows: seqLen * inputSize, cols: 1)
        var dh_next = Matrix(rows: hiddenSize, cols: 1)
        
        for t in stride(from: seqLen - 1, through: 0, by: -1) {
            var dh: Matrix
            if returnSequences {
                let startIdx = t * hiddenSize
                dh = Matrix(rows: hiddenSize, cols: 1,
                           data: Array(gradOutput.data[startIdx..<(startIdx + hiddenSize)]))
                dh = Matrix.add(dh, dh_next)
            } else {
                dh = t == seqLen - 1 ? gradOutput : dh_next
            }
            
            let (r, z, n) = cachedGates[t]
            let h_prev = cachedHiddenStates[t]
            let x = cachedInputs[t]
            
            // Gradient through: h = (1-z) * n + z * h_prev
            var dn = Matrix(rows: hiddenSize, cols: 1)
            var dz = Matrix(rows: hiddenSize, cols: 1)
            var dh_prev = Matrix(rows: hiddenSize, cols: 1)
            
            for j in 0..<hiddenSize {
                dn.data[j] = dh.data[j] * (1 - z.data[j])
                dz.data[j] = dh.data[j] * (h_prev.data[j] - n.data[j])
                dh_prev.data[j] = dh.data[j] * z.data[j]
            }
            
            // Gradient through tanh
            for j in 0..<hiddenSize {
                dn.data[j] *= (1 - n.data[j] * n.data[j])
            }
            
            // Gradient through sigmoid for z
            for j in 0..<hiddenSize {
                dz.data[j] *= z.data[j] * (1 - z.data[j])
            }
            
            // Gradient w.r.t. r
            let hGates = Matrix.multiply(weightsHH, h_prev)
            var dr = Matrix(rows: hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                dr.data[j] = dn.data[j] * hGates.data[j + 2 * hiddenSize] * r.data[j] * (1 - r.data[j])
            }
            
            // Concatenate gate gradients
            var dGates = Matrix(rows: 3 * hiddenSize, cols: 1)
            for j in 0..<hiddenSize {
                dGates.data[j] = dr.data[j]
                dGates.data[j + hiddenSize] = dz.data[j]
                dGates.data[j + 2 * hiddenSize] = dn.data[j]
            }
            
            // Accumulate gradients
            let dWih = Matrix.multiply(dGates, x, transposeB: true)
            weightsIHGrad = Matrix.add(weightsIHGrad, dWih)
            
            let dWhh = Matrix.multiply(dGates, h_prev, transposeB: true)
            weightsHHGrad = Matrix.add(weightsHHGrad, dWhh)
            
            biasGrad = Matrix.add(biasGrad, dGates)
            
            // Gradient w.r.t. input
            let dx = Matrix.multiply(weightsIH, dGates, transposeA: true)
            for j in 0..<inputSize {
                gradInput.data[t * inputSize + j] = dx.data[j]
            }
            
            // Gradient w.r.t. previous hidden state
            let dhFromGates = Matrix.multiply(weightsHH, dGates, transposeA: true)
            dh_next = Matrix.add(dh_prev, dhFromGates)
        }
        
        return gradInput
    }
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    public func parameters() -> [Matrix] {
        return [weightsIH, weightsHH, bias]
    }
    
    public func gradients() -> [Matrix] {
        return [weightsIHGrad, weightsHHGrad, biasGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        var scaledIHGrad = weightsIHGrad
        scaledIHGrad.scale(by: learningRate)
        
        var scaledHHGrad = weightsHHGrad
        scaledHHGrad.scale(by: learningRate)
        
        var scaledBiasGrad = biasGrad
        scaledBiasGrad.scale(by: learningRate)
        
        weightsIH = Matrix.subtract(weightsIH, scaledIHGrad)
        weightsHH = Matrix.subtract(weightsHH, scaledHHGrad)
        bias = Matrix.subtract(bias, scaledBiasGrad)
        
        weightsIHGrad.zero()
        weightsHHGrad.zero()
        biasGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        weightsIHGrad.scale(by: scale)
        weightsHHGrad.scale(by: scale)
        biasGrad.scale(by: scale)
    }
}

// MARK: - Embedding Layer

/// Embedding layer for converting integer indices to dense vectors
/// Useful for NLP tasks
public class EmbeddingLayer: Layer {
    /// Vocabulary size
    public let vocabSize: Int
    
    /// Embedding dimension
    public let embeddingDim: Int
    
    /// Embedding weight matrix [vocabSize, embeddingDim]
    public var embeddings: Matrix
    
    /// Gradient w.r.t. embeddings
    private var embeddingGrad: Matrix
    
    /// Cached indices for backward pass
    private var cachedIndices: [Int] = []
    
    /// Initialize embedding layer
    public init(vocabSize: Int, embeddingDim: Int) {
        precondition(vocabSize > 0, "Vocabulary size must be positive")
        precondition(embeddingDim > 0, "Embedding dimension must be positive")
        
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        
        // Initialize with small random values
        let bound = 1.0 / sqrt(Float(embeddingDim))
        self.embeddings = Matrix(rows: vocabSize, cols: embeddingDim, randomInRange: -bound, bound)
        self.embeddingGrad = Matrix(rows: vocabSize, cols: embeddingDim)
    }
    
    /// Forward pass - looks up embeddings for input indices
    /// Input: (seqLen, 1) with integer values (stored as Float)
    /// Output: (seqLen * embeddingDim, 1)
    public func forward(_ input: Matrix) -> Matrix {
        let seqLen = input.rows
        cachedIndices = []
        
        var output = Matrix(rows: seqLen * embeddingDim, cols: 1)
        
        for i in 0..<seqLen {
            let idx = Int(input.data[i])
            precondition(idx >= 0 && idx < vocabSize, "Index \(idx) out of vocabulary range")
            cachedIndices.append(idx)
            
            for j in 0..<embeddingDim {
                output.data[i * embeddingDim + j] = embeddings[idx, j]
            }
        }
        
        return output
    }
    
    public func backward(_ gradOutput: Matrix) -> Matrix {
        let seqLen = cachedIndices.count
        
        // Accumulate gradients into embedding matrix
        for i in 0..<seqLen {
            let idx = cachedIndices[i]
            for j in 0..<embeddingDim {
                embeddingGrad[idx, j] += gradOutput.data[i * embeddingDim + j]
            }
        }
        
        // Return zeros as gradient w.r.t. indices (not differentiable)
        return Matrix(rows: seqLen, cols: 1)
    }
    
    public func parameters() -> [Matrix] {
        return [embeddings]
    }
    
    public func gradients() -> [Matrix] {
        return [embeddingGrad]
    }
    
    public func updateParameters(learningRate: Float) {
        var scaledGrad = embeddingGrad
        scaledGrad.scale(by: learningRate)
        embeddings = Matrix.subtract(embeddings, scaledGrad)
        embeddingGrad.zero()
    }
    
    public func scaleGradients(by scale: Float) {
        embeddingGrad.scale(by: scale)
    }
}
