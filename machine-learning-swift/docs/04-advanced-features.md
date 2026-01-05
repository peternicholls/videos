# Part 4: Advanced Features

This part covers advanced MLSwift features: custom optimizers, model serialization, and hyperparameter tuning.

## Table of Contents

1. [Optimizers](#optimizers)
2. [Model Serialization](#model-serialization)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Training Best Practices](#training-best-practices)
5. [Next Steps](#next-steps)

---

## Optimizers

Optimizers determine how model parameters are updated during training. MLSwift includes several commonly used optimizers.

### SGD (Stochastic Gradient Descent)

The simplest optimizer—updates weights directly by gradient:

```swift
let optimizer = SGDOptimizer()

// Parameters are updated: w = w - learningRate * gradient
```

**When to use**: Baseline, simple problems, when you want maximum control.

### SGD with Momentum

Adds "velocity" to help escape local minima and smooth updates:

```swift
let optimizer = SGDMomentumOptimizer(momentum: 0.9)

// Velocity: v = momentum * v + gradient
// Update: w = w - learningRate * v
```

**When to use**: Most training scenarios, good default choice.

### Adam (Adaptive Moment Estimation)

Adapts learning rate for each parameter based on gradient history:

```swift
let optimizer = AdamOptimizer(beta1: 0.9, beta2: 0.999)

// Maintains running averages of gradient and squared gradient
// Adapts learning rate per parameter
```

**When to use**: Deep networks, transformer models, when SGD struggles.

### RMSprop

Adapts learning rate using moving average of squared gradients:

```swift
let optimizer = RMSpropOptimizer(decay: 0.9)

// Similar to Adam but without momentum on gradients
```

**When to use**: Recurrent networks, non-stationary problems.

### Using Optimizers

```swift
import MLSwift

let model = SequentialModel()
model.add(DenseLayer(inputSize: 10, outputSize: 20))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 20, outputSize: 1))

let optimizer = AdamOptimizer()
let learningRate: Float = 0.001

for epoch in 1...100 {
    for (input, target) in zip(trainInputs, trainTargets) {
        // Forward pass
        let output = model.forward(input)
        
        // Backward pass
        let gradOutput = Loss.meanSquaredErrorBackward(output, target)
        model.backward(gradOutput)
        
        // Collect parameters and gradients
        var allParams: [Matrix] = []
        var allGrads: [Matrix] = []
        for layer in model.getLayers() {
            allParams.append(contentsOf: layer.parameters())
            allGrads.append(contentsOf: layer.gradients())
        }
        
        // Update with optimizer
        optimizer.update(parameters: &allParams, gradients: allGrads, learningRate: learningRate)
    }
}
```

### Optimizer Comparison

| Optimizer | Memory | Speed | Best For |
|-----------|--------|-------|----------|
| SGD | Low | Fast | Simple problems, fine-tuning |
| SGD+Momentum | Medium | Fast | Most classification tasks |
| Adam | High | Medium | Deep networks, transformers |
| RMSprop | Medium | Medium | RNNs, non-stationary objectives |

## Model Serialization

Save trained models to disk and load them later.

### Saving a Model

```swift
import MLSwift
import Foundation

// Train your model
let model = SequentialModel()
model.add(DenseLayer(inputSize: 784, outputSize: 128))
model.add(ReLULayer())
model.add(DenseLayer(inputSize: 128, outputSize: 10))
model.add(SoftmaxLayer())
// ... training code ...

// Save to file
let saveURL = URL(fileURLWithPath: "trained_model.json")
do {
    try model.save(to: saveURL)
    print("Model saved to: \(saveURL.path)")
} catch {
    print("Save failed: \(error)")
}
```

### Loading a Model

```swift
import MLSwift
import Foundation

let loadURL = URL(fileURLWithPath: "trained_model.json")

do {
    let model = try SequentialModel.load(from: loadURL)
    print("Model loaded successfully!")
    
    // Set inference mode
    for layer in model.getLayers() {
        if let dropout = layer as? DropoutLayer { dropout.training = false }
        if let batchNorm = layer as? BatchNormLayer { batchNorm.training = false }
    }
    
    // Make predictions
    let prediction = model.forward(input)
    print("Prediction: \(prediction.data)")
} catch {
    print("Load failed: \(error)")
}
```

### Model File Format

Models are saved as JSON:

```json
{
  "version": "1.0",
  "savedDate": "2024-01-01T12:00:00Z",
  "metadata": {
    "framework": "MLSwift",
    "platform": "macOS"
  },
  "architecture": [
    {
      "type": "Dense",
      "config": {"inputSize": "784", "outputSize": "128"},
      "parameters": [
        {"rows": 128, "cols": 784, "data": [...]},
        {"rows": 128, "cols": 1, "data": [...]}
      ]
    },
    {"type": "ReLU", "config": {}, "parameters": []},
    ...
  ]
}
```

### Checkpointing During Training

Save checkpoints to prevent data loss:

```swift
for epoch in 1...100 {
    // Training code...
    
    // Save checkpoint every 10 epochs
    if epoch % 10 == 0 {
        let checkpointURL = URL(fileURLWithPath: "checkpoint_\(epoch).json")
        try? model.save(to: checkpointURL)
        print("Checkpoint saved: epoch \(epoch)")
    }
}

// Save final model
try model.save(to: URL(fileURLWithPath: "final_model.json"))
```

### Continuing Training

Load a saved model and continue training:

```swift
// Load previous checkpoint
let model = try SequentialModel.load(from: URL(fileURLWithPath: "checkpoint_50.json"))

// Set loss function (not serialized)
model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)

// Enable training mode
for layer in model.getLayers() {
    if let dropout = layer as? DropoutLayer { dropout.training = true }
    if let batchNorm = layer as? BatchNormLayer { batchNorm.training = true }
}

// Continue training
for epoch in 51...100 {
    for (input, target) in zip(trainInputs, trainTargets) {
        model.trainStep(input: input, target: target, learningRate: 0.001)
    }
}
```

## Hyperparameter Tuning

Hyperparameters are settings you choose before training. Tuning them can significantly improve performance.

### Key Hyperparameters

| Hyperparameter | Typical Range | Effect |
|----------------|---------------|--------|
| Learning Rate | 0.0001 - 0.1 | How fast parameters update |
| Batch Size | 16 - 128 | Samples per gradient update |
| Hidden Size | 32 - 512 | Network capacity |
| Dropout Rate | 0.1 - 0.5 | Regularization strength |
| Epochs | 10 - 100+ | Training duration |

### Learning Rate

The most important hyperparameter:

```swift
// Too high: Training is unstable, loss jumps around
// Too low: Training is very slow
// Just right: Loss decreases smoothly

// Common starting points by task:
let classificationLR: Float = 0.01
let regressionLR: Float = 0.001
let fineTuningLR: Float = 0.0001
```

### Learning Rate Schedule (Manual)

Decrease learning rate during training:

```swift
var learningRate: Float = 0.01

for epoch in 1...100 {
    // Train for this epoch
    for (input, target) in zip(trainInputs, trainTargets) {
        model.trainStep(input: input, target: target, learningRate: learningRate)
    }
    
    // Decay learning rate every 30 epochs
    if epoch % 30 == 0 {
        learningRate *= 0.1
        print("Learning rate reduced to: \(learningRate)")
    }
}
```

### Grid Search

Systematically try combinations:

```swift
let learningRates: [Float] = [0.1, 0.01, 0.001]
let hiddenSizes = [32, 64, 128]
let batchSizes = [16, 32, 64]

var bestAccuracy: Float = 0.0
var bestConfig: (Float, Int, Int) = (0, 0, 0)

for lr in learningRates {
    for hidden in hiddenSizes {
        for batch in batchSizes {
            // Create and train model with these hyperparameters
            let model = createModel(hiddenSize: hidden)
            trainModel(model, learningRate: lr, batchSize: batch, epochs: 10)
            
            let accuracy = evaluate(model, on: validationData)
            
            if accuracy > bestAccuracy {
                bestAccuracy = accuracy
                bestConfig = (lr, hidden, batch)
            }
            
            print("LR=\(lr), Hidden=\(hidden), Batch=\(batch) → Accuracy=\(accuracy)%")
        }
    }
}

print("Best config: LR=\(bestConfig.0), Hidden=\(bestConfig.1), Batch=\(bestConfig.2)")
print("Best accuracy: \(bestAccuracy)%")
```

### Early Stopping

Stop training when validation performance stops improving:

```swift
var bestValLoss: Float = Float.infinity
var patienceCounter = 0
let patience = 5  // Stop after 5 epochs without improvement

for epoch in 1...100 {
    // Training
    trainOneEpoch(model, data: trainData, learningRate: learningRate)
    
    // Validation
    let valLoss = computeLoss(model, on: validationData)
    
    if valLoss < bestValLoss {
        bestValLoss = valLoss
        patienceCounter = 0
        try model.save(to: URL(fileURLWithPath: "best_model.json"))
        print("Epoch \(epoch): New best! Val Loss = \(valLoss)")
    } else {
        patienceCounter += 1
        print("Epoch \(epoch): Val Loss = \(valLoss) (patience: \(patienceCounter)/\(patience))")
    }
    
    if patienceCounter >= patience {
        print("Early stopping at epoch \(epoch)")
        break
    }
}

// Load best model
let bestModel = try SequentialModel.load(from: URL(fileURLWithPath: "best_model.json"))
```

## Training Best Practices

### 1. Data Preparation

```swift
// Normalize inputs to [0, 1] or [-1, 1]
let normalizedInputs = inputs.map { minMaxNormalize($0) }

// Shuffle before training
let (shuffledInputs, shuffledTargets) = shuffleDataset(inputs: inputs, targets: targets)

// Split into train/validation/test
let (trainData, valData) = trainValidationSplit(data: allData, splitRatio: 0.8)
```

### 2. Weight Initialization

MLSwift initializes weights automatically, but understanding helps debugging:

```swift
// Dense layers use Xavier/Glorot initialization
// Good for sigmoid/tanh activations

// For ReLU, He initialization is better (MLSwift handles this)
```

### 3. Monitoring Training

```swift
for epoch in 1...epochs {
    var trainLoss: Float = 0.0
    
    for (input, target) in zip(trainInputs, trainTargets) {
        trainLoss += model.trainStep(input: input, target: target, learningRate: lr)
    }
    
    let avgTrainLoss = trainLoss / Float(trainInputs.count)
    let valAccuracy = computeAccuracy(model: model, inputs: valInputs, targets: valTargets)
    
    // Watch for:
    // - Train loss not decreasing: LR too low or model too simple
    // - Train loss jumping: LR too high
    // - Val accuracy not improving: Overfitting, add regularization
    // - Both losses high: Underfitting, increase model capacity
    
    print("Epoch \(epoch): Train Loss = \(String(format: "%.4f", avgTrainLoss)), Val Acc = \(String(format: "%.1f", valAccuracy))%")
}
```

### 4. Debugging Tips

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Loss is NaN | LR too high, exploding gradients | Reduce LR, add gradient clipping |
| Loss doesn't decrease | LR too low, bad initialization | Increase LR, check data |
| Training acc high, val acc low | Overfitting | Add dropout, reduce capacity |
| Both accuracies low | Underfitting | Increase capacity, train longer |

## Complete Example: Best Practices

```swift
import MLSwift
import Foundation

func trainWithBestPractices() throws {
    // 1. Prepare data
    print("Preparing data...")
    let (trainInputs, trainTargets, valInputs, valTargets) = prepareData()
    
    // 2. Create model
    print("Creating model...")
    let model = SequentialModel()
    model.add(DenseLayer(inputSize: 784, outputSize: 256))
    model.add(BatchNormLayer(numFeatures: 256))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.4))
    model.add(DenseLayer(inputSize: 256, outputSize: 128))
    model.add(BatchNormLayer(numFeatures: 128))
    model.add(ReLULayer())
    model.add(DropoutLayer(dropoutRate: 0.3))
    model.add(DenseLayer(inputSize: 128, outputSize: 10))
    model.add(SoftmaxLayer())
    model.setLoss(Loss.crossEntropy, gradient: Loss.crossEntropyBackward)
    
    // 3. Training settings
    var learningRate: Float = 0.01
    let epochs = 50
    let patience = 5
    var bestValAcc: Float = 0.0
    var patienceCounter = 0
    
    // 4. Training loop
    print("Training...")
    for epoch in 1...epochs {
        // Shuffle data each epoch
        let (sInputs, sTargets) = shuffleDataset(inputs: trainInputs, targets: trainTargets)
        
        // Train
        var trainLoss: Float = 0.0
        for (input, target) in zip(sInputs, sTargets) {
            trainLoss += model.trainStep(input: input, target: target, learningRate: learningRate)
        }
        
        // Evaluate (disable dropout/batchnorm training)
        setInferenceMode(model)
        let valAcc = computeAccuracy(model: model, inputs: valInputs, targets: valTargets)
        setTrainingMode(model)
        
        // Learning rate decay
        if epoch % 20 == 0 {
            learningRate *= 0.5
        }
        
        // Early stopping check
        if valAcc > bestValAcc {
            bestValAcc = valAcc
            patienceCounter = 0
            try model.save(to: URL(fileURLWithPath: "best_model.json"))
        } else {
            patienceCounter += 1
        }
        
        print("Epoch \(epoch): Loss=\(String(format: "%.4f", trainLoss/Float(trainInputs.count))), ValAcc=\(String(format: "%.1f", valAcc))%")
        
        if patienceCounter >= patience {
            print("Early stopping!")
            break
        }
    }
    
    // 5. Load best model for final evaluation
    let finalModel = try SequentialModel.load(from: URL(fileURLWithPath: "best_model.json"))
    setInferenceMode(finalModel)
    let finalAcc = computeAccuracy(model: finalModel, inputs: valInputs, targets: valTargets)
    print("Final Validation Accuracy: \(String(format: "%.1f", finalAcc))%")
}

func setInferenceMode(_ model: SequentialModel) {
    for layer in model.getLayers() {
        if let d = layer as? DropoutLayer { d.training = false }
        if let b = layer as? BatchNormLayer { b.training = false }
    }
}

func setTrainingMode(_ model: SequentialModel) {
    for layer in model.getLayers() {
        if let d = layer as? DropoutLayer { d.training = true }
        if let b = layer as? BatchNormLayer { b.training = true }
    }
}
```

## Next Steps

Continue to [Part 5: Apple Framework Integration](05-apple-integration.md) to learn about:
- Metal compute shaders in depth
- Accelerate framework integration
- CoreML export for deployment

---

[← Part 3: Building Neural Networks](03-building-networks.md) | [Part 5: Apple Framework Integration →](05-apple-integration.md)
