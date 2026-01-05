/// Visualization.swift
/// Visualization and debugging tools for neural networks
/// Provides model summary, training metrics logging, and more

import Foundation

// MARK: - Model Summary

extension SequentialModel {
    
    /// Print a summary of the model architecture
    /// Similar to Keras model.summary()
    public func summary() {
        print("=" * 70)
        print("Model Summary")
        print("=" * 70)
        
        let headerFormat = "%-25s %-20s %-15s"
        print(String(format: headerFormat, "Layer (type)", "Output Shape", "Param #"))
        print("-" * 70)
        
        var totalParams = 0
        var trainableParams = 0
        var prevOutputSize = 0
        
        for (index, layer) in getLayers().enumerated() {
            let layerInfo = getLayerInfo(layer, index: index, prevOutputSize: &prevOutputSize)
            print(String(format: "%-25s %-20s %-15s", 
                        layerInfo.name,
                        layerInfo.outputShape,
                        formatNumber(layerInfo.paramCount)))
            
            totalParams += layerInfo.paramCount
            trainableParams += layerInfo.trainableCount
        }
        
        print("=" * 70)
        print("Total params: \(formatNumber(totalParams))")
        print("Trainable params: \(formatNumber(trainableParams))")
        print("Non-trainable params: \(formatNumber(totalParams - trainableParams))")
        print("=" * 70)
    }
    
    /// Get information about a specific layer
    private func getLayerInfo(_ layer: Layer, index: Int, prevOutputSize: inout Int) -> LayerInfo {
        var name = String(describing: type(of: layer))
        var outputShape = ""
        var paramCount = 0
        var trainableCount = 0
        
        switch layer {
        case let dense as DenseLayer:
            name = "dense_\(index) (Dense)"
            let outputSize = dense.weights.rows
            outputShape = "(\(outputSize),)"
            paramCount = dense.weights.count + dense.bias.count
            trainableCount = paramCount
            prevOutputSize = outputSize
            
        case _ as ReLULayer:
            name = "relu_\(index) (ReLU)"
            outputShape = "(\(prevOutputSize),)"
            
        case _ as SigmoidLayer:
            name = "sigmoid_\(index) (Sigmoid)"
            outputShape = "(\(prevOutputSize),)"
            
        case _ as TanhLayer:
            name = "tanh_\(index) (Tanh)"
            outputShape = "(\(prevOutputSize),)"
            
        case _ as SoftmaxLayer:
            name = "softmax_\(index) (Softmax)"
            outputShape = "(\(prevOutputSize),)"
            
        case let dropout as DropoutLayer:
            name = "dropout_\(index) (Dropout)"
            outputShape = "(\(prevOutputSize),)"
            
        case let batchNorm as BatchNormLayer:
            name = "batchnorm_\(index) (BatchNorm)"
            outputShape = "(\(prevOutputSize),)"
            paramCount = batchNorm.gamma.count + batchNorm.beta.count
            trainableCount = paramCount
            
        case let conv2d as Conv2DLayer:
            name = "conv2d_\(index) (Conv2D)"
            outputShape = "(\(conv2d.outputChannels), ?, ?)"
            paramCount = conv2d.weights.count + conv2d.bias.count
            trainableCount = paramCount
            
        case let maxPool as MaxPool2DLayer:
            name = "maxpool_\(index) (MaxPool2D)"
            outputShape = "(?, ?, ?)"
            
        case _ as FlattenLayer:
            name = "flatten_\(index) (Flatten)"
            outputShape = "(\(prevOutputSize),)"
            
        case let lstm as LSTMLayer:
            name = "lstm_\(index) (LSTM)"
            outputShape = lstm.returnSequences ? "(seq, \(lstm.hiddenSize))" : "(\(lstm.hiddenSize),)"
            paramCount = lstm.weightsIH.count + lstm.weightsHH.count + lstm.bias.count
            trainableCount = paramCount
            prevOutputSize = lstm.hiddenSize
            
        case let gru as GRULayer:
            name = "gru_\(index) (GRU)"
            outputShape = gru.returnSequences ? "(seq, \(gru.hiddenSize))" : "(\(gru.hiddenSize),)"
            paramCount = gru.weightsIH.count + gru.weightsHH.count + gru.bias.count
            trainableCount = paramCount
            prevOutputSize = gru.hiddenSize
            
        case let embedding as EmbeddingLayer:
            name = "embedding_\(index) (Embedding)"
            outputShape = "(seq, \(embedding.embeddingDim))"
            paramCount = embedding.embeddings.count
            trainableCount = paramCount
            prevOutputSize = embedding.embeddingDim
            
        default:
            name = "layer_\(index)"
            outputShape = "(?)"
        }
        
        return LayerInfo(name: name, outputShape: outputShape, paramCount: paramCount, trainableCount: trainableCount)
    }
    
    /// Format large numbers with commas
    private func formatNumber(_ num: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: num)) ?? String(num)
    }
}

/// Layer information for summary
private struct LayerInfo {
    let name: String
    let outputShape: String
    let paramCount: Int
    let trainableCount: Int
}

/// String multiplication helper
private func *(lhs: String, rhs: Int) -> String {
    return String(repeating: lhs, count: rhs)
}

// MARK: - Training History

/// Records training metrics over epochs
public class TrainingHistory {
    /// Loss values per epoch
    public private(set) var trainLoss: [Float] = []
    
    /// Validation loss per epoch
    public private(set) var valLoss: [Float] = []
    
    /// Training accuracy per epoch
    public private(set) var trainAccuracy: [Float] = []
    
    /// Validation accuracy per epoch
    public private(set) var valAccuracy: [Float] = []
    
    /// Learning rates used per epoch
    public private(set) var learningRates: [Float] = []
    
    /// Gradient norms per epoch
    public private(set) var gradientNorms: [Float] = []
    
    /// Epoch timestamps
    public private(set) var timestamps: [Date] = []
    
    /// Custom metrics
    public private(set) var customMetrics: [String: [Float]] = [:]
    
    public init() {}
    
    /// Record metrics for an epoch
    /// - Parameters:
    ///   - epoch: Epoch number
    ///   - trainLoss: Training loss
    ///   - valLoss: Validation loss (optional)
    ///   - trainAccuracy: Training accuracy (optional)
    ///   - valAccuracy: Validation accuracy (optional)
    ///   - learningRate: Learning rate used (optional)
    ///   - gradientNorm: Gradient norm (optional)
    public func record(
        epoch: Int,
        trainLoss: Float,
        valLoss: Float? = nil,
        trainAccuracy: Float? = nil,
        valAccuracy: Float? = nil,
        learningRate: Float? = nil,
        gradientNorm: Float? = nil
    ) {
        self.trainLoss.append(trainLoss)
        self.valLoss.append(valLoss ?? 0)
        self.trainAccuracy.append(trainAccuracy ?? 0)
        self.valAccuracy.append(valAccuracy ?? 0)
        self.learningRates.append(learningRate ?? 0)
        self.gradientNorms.append(gradientNorm ?? 0)
        self.timestamps.append(Date())
    }
    
    /// Record a custom metric
    public func recordCustom(name: String, value: Float) {
        if customMetrics[name] == nil {
            customMetrics[name] = []
        }
        customMetrics[name]?.append(value)
    }
    
    /// Get the best epoch based on validation loss
    public func bestEpoch() -> Int? {
        guard !valLoss.isEmpty else { return nil }
        return valLoss.enumerated().min(by: { $0.element < $1.element })?.offset
    }
    
    /// Print training summary
    public func summary() {
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        
        print("Total epochs: \(trainLoss.count)")
        
        if let best = bestEpoch() {
            print("Best epoch: \(best + 1)")
            print("  Train Loss: \(String(format: "%.4f", trainLoss[best]))")
            if valLoss[best] > 0 {
                print("  Val Loss: \(String(format: "%.4f", valLoss[best]))")
            }
            if valAccuracy[best] > 0 {
                print("  Val Accuracy: \(String(format: "%.2f%%", valAccuracy[best] * 100))")
            }
        }
        
        if let finalTrainLoss = trainLoss.last {
            print("\nFinal epoch:")
            print("  Train Loss: \(String(format: "%.4f", finalTrainLoss))")
        }
        if let finalValLoss = valLoss.last, finalValLoss > 0 {
            print("  Val Loss: \(String(format: "%.4f", finalValLoss))")
        }
        if let finalAccuracy = valAccuracy.last, finalAccuracy > 0 {
            print("  Val Accuracy: \(String(format: "%.2f%%", finalAccuracy * 100))")
        }
        
        if let firstTime = timestamps.first, let lastTime = timestamps.last {
            let duration = lastTime.timeIntervalSince(firstTime)
            print("\nTotal training time: \(formatDuration(duration))")
            if trainLoss.count > 1 {
                let avgTimePerEpoch = duration / Double(trainLoss.count - 1)
                print("Average time per epoch: \(String(format: "%.1f", avgTimePerEpoch))s")
            }
        }
        
        print("=" * 60)
    }
    
    /// Format duration in human-readable form
    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.1f seconds", seconds)
        } else if seconds < 3600 {
            let minutes = Int(seconds / 60)
            let secs = Int(seconds.truncatingRemainder(dividingBy: 60))
            return "\(minutes)m \(secs)s"
        } else {
            let hours = Int(seconds / 3600)
            let minutes = Int((seconds / 60).truncatingRemainder(dividingBy: 60))
            return "\(hours)h \(minutes)m"
        }
    }
    
    /// Export history to CSV format
    public func toCSV() -> String {
        var csv = "epoch,train_loss,val_loss,train_acc,val_acc,learning_rate,grad_norm\n"
        
        for i in 0..<trainLoss.count {
            csv += "\(i + 1),"
            csv += "\(trainLoss[i]),"
            csv += "\(valLoss.count > i ? valLoss[i] : 0),"
            csv += "\(trainAccuracy.count > i ? trainAccuracy[i] : 0),"
            csv += "\(valAccuracy.count > i ? valAccuracy[i] : 0),"
            csv += "\(learningRates.count > i ? learningRates[i] : 0),"
            csv += "\(gradientNorms.count > i ? gradientNorms[i] : 0)\n"
        }
        
        return csv
    }
    
    /// Save history to a CSV file
    public func save(to url: URL) throws {
        let csv = toCSV()
        try csv.write(to: url, atomically: true, encoding: .utf8)
    }
}

// MARK: - Progress Logger

/// Training progress logger with various output formats
public class TrainingLogger {
    /// Logging verbosity level
    public enum Verbosity {
        case silent    // No output
        case minimal   // Only epoch summaries
        case normal    // Epoch summaries and progress
        case verbose   // All details including batch progress
    }
    
    /// Current verbosity level
    public var verbosity: Verbosity
    
    /// Training history
    public var history: TrainingHistory
    
    /// Progress bar width
    public var progressBarWidth: Int = 30
    
    /// Start time
    private var startTime: Date?
    
    /// Current epoch start time
    private var epochStartTime: Date?
    
    public init(verbosity: Verbosity = .normal) {
        self.verbosity = verbosity
        self.history = TrainingHistory()
    }
    
    /// Called at the start of training
    public func onTrainingStart(totalEpochs: Int) {
        startTime = Date()
        if verbosity != .silent {
            print("\nStarting training for \(totalEpochs) epochs...")
            print("-" * 60)
        }
    }
    
    /// Called at the start of each epoch
    public func onEpochStart(epoch: Int, totalEpochs: Int) {
        epochStartTime = Date()
        if verbosity == .verbose {
            print("\nEpoch \(epoch)/\(totalEpochs)")
        }
    }
    
    /// Called during training to show batch progress
    public func onBatchEnd(batch: Int, totalBatches: Int, batchLoss: Float) {
        if verbosity == .verbose {
            let progress = Float(batch) / Float(totalBatches)
            let filledWidth = Int(progress * Float(progressBarWidth))
            let emptyWidth = progressBarWidth - filledWidth
            
            let bar = "[" + String(repeating: "=", count: filledWidth) + 
                     String(repeating: " ", count: emptyWidth) + "]"
            
            print("\r\(bar) \(batch)/\(totalBatches) - loss: \(String(format: "%.4f", batchLoss))", terminator: "")
            fflush(stdout)
        }
    }
    
    /// Called at the end of each epoch
    public func onEpochEnd(
        epoch: Int,
        totalEpochs: Int,
        trainLoss: Float,
        valLoss: Float? = nil,
        trainAccuracy: Float? = nil,
        valAccuracy: Float? = nil,
        learningRate: Float? = nil
    ) {
        history.record(
            epoch: epoch,
            trainLoss: trainLoss,
            valLoss: valLoss,
            trainAccuracy: trainAccuracy,
            valAccuracy: valAccuracy,
            learningRate: learningRate
        )
        
        if verbosity == .silent { return }
        
        if verbosity == .verbose {
            print()  // New line after progress bar
        }
        
        var message = "Epoch \(epoch)/\(totalEpochs)"
        
        // Add epoch duration
        if let epochStart = epochStartTime {
            let duration = Date().timeIntervalSince(epochStart)
            message += " - \(String(format: "%.1f", duration))s"
        }
        
        message += " - loss: \(String(format: "%.4f", trainLoss))"
        
        if let valLoss = valLoss {
            message += " - val_loss: \(String(format: "%.4f", valLoss))"
        }
        
        if let trainAcc = trainAccuracy {
            message += " - acc: \(String(format: "%.2f%%", trainAcc * 100))"
        }
        
        if let valAcc = valAccuracy {
            message += " - val_acc: \(String(format: "%.2f%%", valAcc * 100))"
        }
        
        if let lr = learningRate {
            message += " - lr: \(String(format: "%.2e", lr))"
        }
        
        print(message)
    }
    
    /// Called at the end of training
    public func onTrainingEnd() {
        if verbosity != .silent {
            history.summary()
        }
    }
}

// MARK: - Confusion Matrix

/// Confusion matrix for classification evaluation
public class ConfusionMatrix {
    /// Matrix data [actual][predicted]
    public private(set) var matrix: [[Int]]
    
    /// Number of classes
    public let numClasses: Int
    
    /// Class labels (optional)
    public var labels: [String]?
    
    public init(numClasses: Int, labels: [String]? = nil) {
        self.numClasses = numClasses
        self.matrix = [[Int]](repeating: [Int](repeating: 0, count: numClasses), count: numClasses)
        self.labels = labels
    }
    
    /// Add a prediction to the matrix
    public func add(actual: Int, predicted: Int) {
        precondition(actual >= 0 && actual < numClasses, "Actual class out of range")
        precondition(predicted >= 0 && predicted < numClasses, "Predicted class out of range")
        matrix[actual][predicted] += 1
    }
    
    /// Compute accuracy
    public func accuracy() -> Float {
        let correct = (0..<numClasses).map { matrix[$0][$0] }.reduce(0, +)
        let total = matrix.flatMap { $0 }.reduce(0, +)
        return total > 0 ? Float(correct) / Float(total) : 0
    }
    
    /// Compute precision for a class
    public func precision(forClass c: Int) -> Float {
        let predictedPositive = (0..<numClasses).map { matrix[$0][c] }.reduce(0, +)
        let truePositive = matrix[c][c]
        return predictedPositive > 0 ? Float(truePositive) / Float(predictedPositive) : 0
    }
    
    /// Compute recall for a class
    public func recall(forClass c: Int) -> Float {
        let actualPositive = matrix[c].reduce(0, +)
        let truePositive = matrix[c][c]
        return actualPositive > 0 ? Float(truePositive) / Float(actualPositive) : 0
    }
    
    /// Compute F1 score for a class
    public func f1Score(forClass c: Int) -> Float {
        let p = precision(forClass: c)
        let r = recall(forClass: c)
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0
    }
    
    /// Compute macro-averaged F1 score
    public func macroF1() -> Float {
        let scores = (0..<numClasses).map { f1Score(forClass: $0) }
        return scores.reduce(0, +) / Float(numClasses)
    }
    
    /// Print the confusion matrix
    public func print() {
        let width = 8
        
        Swift.print("\nConfusion Matrix:")
        Swift.print("-" * (width * (numClasses + 1) + numClasses + 1))
        
        // Header
        var header = String(repeating: " ", count: width) + "|"
        for i in 0..<numClasses {
            let label = labels?[i] ?? "C\(i)"
            header += String(format: "%\(width)s|", label)
        }
        Swift.print(header)
        Swift.print("-" * (width * (numClasses + 1) + numClasses + 1))
        
        // Rows
        for i in 0..<numClasses {
            let label = labels?[i] ?? "C\(i)"
            var row = String(format: "%\(width)s|", label)
            for j in 0..<numClasses {
                row += String(format: "%\(width)d|", matrix[i][j])
            }
            Swift.print(row)
        }
        
        Swift.print("-" * (width * (numClasses + 1) + numClasses + 1))
        Swift.print("\nAccuracy: \(String(format: "%.2f%%", accuracy() * 100))")
        Swift.print("Macro F1: \(String(format: "%.4f", macroF1()))")
        
        // Per-class metrics
        Swift.print("\nPer-class metrics:")
        for i in 0..<numClasses {
            let label = labels?[i] ?? "Class \(i)"
            Swift.print("  \(label): P=\(String(format: "%.2f", precision(forClass: i))), " +
                       "R=\(String(format: "%.2f", recall(forClass: i))), " +
                       "F1=\(String(format: "%.2f", f1Score(forClass: i)))")
        }
    }
    
    /// Reset the matrix
    public func reset() {
        matrix = [[Int]](repeating: [Int](repeating: 0, count: numClasses), count: numClasses)
    }
}

// MARK: - Model Evaluation Utilities

extension SequentialModel {
    
    /// Evaluate model and return confusion matrix
    public func evaluateWithConfusionMatrix(
        inputs: [Matrix],
        targets: [Matrix],
        numClasses: Int,
        labels: [String]? = nil
    ) -> ConfusionMatrix {
        let cm = ConfusionMatrix(numClasses: numClasses, labels: labels)
        
        for (input, target) in zip(inputs, targets) {
            let output = forward(input)
            let predicted = output.argmax()
            let actual = target.argmax()
            cm.add(actual: actual, predicted: predicted)
        }
        
        return cm
    }
}
