/// DataAugmentation.swift
/// Image data augmentation utilities for improving model generalization
/// Optimized for Apple Silicon

import Foundation

/// Data augmentation pipeline for image data
public class DataAugmentation {
    
    /// Augmentation operations to apply
    private var operations: [AugmentationOperation] = []
    
    public init() {}
    
    // MARK: - Builder Pattern Methods
    
    /// Add random horizontal flip
    /// - Parameter probability: Probability of flipping (default 0.5)
    /// - Returns: Self for chaining
    @discardableResult
    public func randomHorizontalFlip(probability: Float = 0.5) -> DataAugmentation {
        operations.append(.horizontalFlip(probability: probability))
        return self
    }
    
    /// Add random vertical flip
    /// - Parameter probability: Probability of flipping (default 0.5)
    /// - Returns: Self for chaining
    @discardableResult
    public func randomVerticalFlip(probability: Float = 0.5) -> DataAugmentation {
        operations.append(.verticalFlip(probability: probability))
        return self
    }
    
    /// Add random rotation
    /// - Parameter maxDegrees: Maximum rotation angle in degrees
    /// - Returns: Self for chaining
    @discardableResult
    public func randomRotation(maxDegrees: Float) -> DataAugmentation {
        operations.append(.rotation(maxDegrees: maxDegrees))
        return self
    }
    
    /// Add random brightness adjustment
    /// - Parameter range: Range for brightness factor (e.g., 0.2 means 0.8 to 1.2)
    /// - Returns: Self for chaining
    @discardableResult
    public func randomBrightness(range: Float) -> DataAugmentation {
        operations.append(.brightness(range: range))
        return self
    }
    
    /// Add random contrast adjustment
    /// - Parameter range: Range for contrast factor
    /// - Returns: Self for chaining
    @discardableResult
    public func randomContrast(range: Float) -> DataAugmentation {
        operations.append(.contrast(range: range))
        return self
    }
    
    /// Add Gaussian noise
    /// - Parameter stdDev: Standard deviation of noise
    /// - Returns: Self for chaining
    @discardableResult
    public func randomNoise(stdDev: Float) -> DataAugmentation {
        operations.append(.noise(stdDev: stdDev))
        return self
    }
    
    /// Add random zoom
    /// - Parameter range: Zoom range (e.g., 0.1 means 0.9 to 1.1)
    /// - Returns: Self for chaining
    @discardableResult
    public func randomZoom(range: Float) -> DataAugmentation {
        operations.append(.zoom(range: range))
        return self
    }
    
    /// Add random crop
    /// - Parameters:
    ///   - outputHeight: Output crop height
    ///   - outputWidth: Output crop width
    /// - Returns: Self for chaining
    @discardableResult
    public func randomCrop(outputHeight: Int, outputWidth: Int) -> DataAugmentation {
        operations.append(.crop(outputHeight: outputHeight, outputWidth: outputWidth))
        return self
    }
    
    /// Add random erasing (cutout)
    /// - Parameters:
    ///   - probability: Probability of applying erasing
    ///   - scale: Range of proportion of area to erase (min, max)
    ///   - ratio: Range of aspect ratio (min, max)
    /// - Returns: Self for chaining
    @discardableResult
    public func randomErasing(probability: Float = 0.5, scale: (Float, Float) = (0.02, 0.33), ratio: (Float, Float) = (0.3, 3.3)) -> DataAugmentation {
        operations.append(.erasing(probability: probability, scale: scale, ratio: ratio))
        return self
    }
    
    /// Add normalization
    /// - Parameters:
    ///   - mean: Mean values (per channel or single value)
    ///   - std: Standard deviation values (per channel or single value)
    /// - Returns: Self for chaining
    @discardableResult
    public func normalize(mean: [Float], std: [Float]) -> DataAugmentation {
        operations.append(.normalize(mean: mean, std: std))
        return self
    }
    
    // MARK: - Apply Augmentation
    
    /// Apply all augmentation operations to an image
    /// - Parameters:
    ///   - image: Input image as Matrix (channels * height * width, 1)
    ///   - height: Image height
    ///   - width: Image width
    ///   - channels: Number of channels (1 for grayscale, 3 for RGB)
    /// - Returns: Augmented image
    public func apply(to image: Matrix, height: Int, width: Int, channels: Int = 1) -> Matrix {
        var result = image
        
        for operation in operations {
            result = applyOperation(operation, to: result, height: height, width: width, channels: channels)
        }
        
        return result
    }
    
    /// Apply augmentation to a batch of images
    public func applyToBatch(_ images: [Matrix], height: Int, width: Int, channels: Int = 1) -> [Matrix] {
        return images.map { apply(to: $0, height: height, width: width, channels: channels) }
    }
    
    // MARK: - Operation Implementations
    
    private func applyOperation(
        _ operation: AugmentationOperation,
        to image: Matrix,
        height: Int,
        width: Int,
        channels: Int
    ) -> Matrix {
        switch operation {
        case .horizontalFlip(let probability):
            return Float.random(in: 0..<1) < probability ? 
                   horizontalFlip(image, height: height, width: width, channels: channels) : image
            
        case .verticalFlip(let probability):
            return Float.random(in: 0..<1) < probability ?
                   verticalFlip(image, height: height, width: width, channels: channels) : image
            
        case .rotation(let maxDegrees):
            let angle = Float.random(in: -maxDegrees...maxDegrees)
            return rotate(image, angle: angle, height: height, width: width, channels: channels)
            
        case .brightness(let range):
            let factor = Float.random(in: (1 - range)...(1 + range))
            return adjustBrightness(image, factor: factor)
            
        case .contrast(let range):
            let factor = Float.random(in: (1 - range)...(1 + range))
            return adjustContrast(image, factor: factor)
            
        case .noise(let stdDev):
            return addNoise(image, stdDev: stdDev)
            
        case .zoom(let range):
            let factor = Float.random(in: (1 - range)...(1 + range))
            return zoom(image, factor: factor, height: height, width: width, channels: channels)
            
        case .crop(let outH, let outW):
            return randomCrop(image, height: height, width: width, channels: channels, outHeight: outH, outWidth: outW)
            
        case .erasing(let probability, let scale, let ratio):
            return Float.random(in: 0..<1) < probability ?
                   randomErase(image, height: height, width: width, channels: channels, scale: scale, ratio: ratio) : image
            
        case .normalize(let mean, let std):
            return normalize(image, mean: mean, std: std, channels: channels)
        }
    }
}

// MARK: - Augmentation Operations Enum

private enum AugmentationOperation {
    case horizontalFlip(probability: Float)
    case verticalFlip(probability: Float)
    case rotation(maxDegrees: Float)
    case brightness(range: Float)
    case contrast(range: Float)
    case noise(stdDev: Float)
    case zoom(range: Float)
    case crop(outputHeight: Int, outputWidth: Int)
    case erasing(probability: Float, scale: (Float, Float), ratio: (Float, Float))
    case normalize(mean: [Float], std: [Float])
}

// MARK: - Individual Augmentation Functions

extension DataAugmentation {
    
    /// Flip image horizontally
    private func horizontalFlip(_ image: Matrix, height: Int, width: Int, channels: Int) -> Matrix {
        var result = image
        
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<(width / 2) {
                    let idx1 = c * height * width + h * width + w
                    let idx2 = c * height * width + h * width + (width - 1 - w)
                    let temp = result.data[idx1]
                    result.data[idx1] = result.data[idx2]
                    result.data[idx2] = temp
                }
            }
        }
        
        return result
    }
    
    /// Flip image vertically
    private func verticalFlip(_ image: Matrix, height: Int, width: Int, channels: Int) -> Matrix {
        var result = image
        
        for c in 0..<channels {
            for h in 0..<(height / 2) {
                for w in 0..<width {
                    let idx1 = c * height * width + h * width + w
                    let idx2 = c * height * width + (height - 1 - h) * width + w
                    let temp = result.data[idx1]
                    result.data[idx1] = result.data[idx2]
                    result.data[idx2] = temp
                }
            }
        }
        
        return result
    }
    
    /// Rotate image by angle in degrees (bilinear interpolation)
    private func rotate(_ image: Matrix, angle: Float, height: Int, width: Int, channels: Int) -> Matrix {
        let radians = angle * Float.pi / 180.0
        let cosA = cos(radians)
        let sinA = sin(radians)
        
        let centerY = Float(height) / 2.0
        let centerX = Float(width) / 2.0
        
        var result = Matrix(rows: image.rows, cols: 1)
        
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    // Translate to center
                    let y = Float(h) - centerY
                    let x = Float(w) - centerX
                    
                    // Rotate
                    let srcY = y * cosA + x * sinA + centerY
                    let srcX = -y * sinA + x * cosA + centerX
                    
                    // Bilinear interpolation
                    let value = bilinearInterpolate(image, c: c, y: srcY, x: srcX, height: height, width: width)
                    
                    let dstIdx = c * height * width + h * width + w
                    result.data[dstIdx] = value
                }
            }
        }
        
        return result
    }
    
    /// Bilinear interpolation for sub-pixel sampling
    private func bilinearInterpolate(_ image: Matrix, c: Int, y: Float, x: Float, height: Int, width: Int) -> Float {
        let y0 = Int(floor(y))
        let y1 = y0 + 1
        let x0 = Int(floor(x))
        let x1 = x0 + 1
        
        let dy = y - Float(y0)
        let dx = x - Float(x0)
        
        func getValue(_ h: Int, _ w: Int) -> Float {
            if h < 0 || h >= height || w < 0 || w >= width {
                return 0.0  // Zero padding for out of bounds
            }
            return image.data[c * height * width + h * width + w]
        }
        
        let v00 = getValue(y0, x0)
        let v01 = getValue(y0, x1)
        let v10 = getValue(y1, x0)
        let v11 = getValue(y1, x1)
        
        return (1 - dy) * (1 - dx) * v00 +
               (1 - dy) * dx * v01 +
               dy * (1 - dx) * v10 +
               dy * dx * v11
    }
    
    /// Adjust brightness
    private func adjustBrightness(_ image: Matrix, factor: Float) -> Matrix {
        var result = image
        for i in 0..<result.data.count {
            result.data[i] = max(0, min(1, result.data[i] * factor))
        }
        return result
    }
    
    /// Adjust contrast
    private func adjustContrast(_ image: Matrix, factor: Float) -> Matrix {
        // Compute mean
        let mean = image.data.reduce(0, +) / Float(image.data.count)
        
        var result = image
        for i in 0..<result.data.count {
            result.data[i] = max(0, min(1, (result.data[i] - mean) * factor + mean))
        }
        return result
    }
    
    /// Add Gaussian noise
    private func addNoise(_ image: Matrix, stdDev: Float) -> Matrix {
        var result = image
        for i in 0..<result.data.count {
            // Box-Muller transform for Gaussian noise
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            let noise = stdDev * sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * Float.pi * u2)
            result.data[i] = max(0, min(1, result.data[i] + noise))
        }
        return result
    }
    
    /// Zoom image
    private func zoom(_ image: Matrix, factor: Float, height: Int, width: Int, channels: Int) -> Matrix {
        let centerY = Float(height) / 2.0
        let centerX = Float(width) / 2.0
        
        var result = Matrix(rows: image.rows, cols: 1)
        
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let srcY = (Float(h) - centerY) / factor + centerY
                    let srcX = (Float(w) - centerX) / factor + centerX
                    
                    let value = bilinearInterpolate(image, c: c, y: srcY, x: srcX, height: height, width: width)
                    
                    let dstIdx = c * height * width + h * width + w
                    result.data[dstIdx] = value
                }
            }
        }
        
        return result
    }
    
    /// Random crop
    private func randomCrop(_ image: Matrix, height: Int, width: Int, channels: Int, outHeight: Int, outWidth: Int) -> Matrix {
        precondition(outHeight <= height && outWidth <= width, "Output dimensions must be smaller than input")
        
        let startH = Int.random(in: 0...(height - outHeight))
        let startW = Int.random(in: 0...(width - outWidth))
        
        var result = Matrix(rows: channels * outHeight * outWidth, cols: 1)
        
        for c in 0..<channels {
            for h in 0..<outHeight {
                for w in 0..<outWidth {
                    let srcIdx = c * height * width + (startH + h) * width + (startW + w)
                    let dstIdx = c * outHeight * outWidth + h * outWidth + w
                    result.data[dstIdx] = image.data[srcIdx]
                }
            }
        }
        
        return result
    }
    
    /// Random erasing (cutout)
    private func randomErase(_ image: Matrix, height: Int, width: Int, channels: Int, scale: (Float, Float), ratio: (Float, Float)) -> Matrix {
        var result = image
        let area = Float(height * width)
        
        // Try to find a valid erasing area
        for _ in 0..<10 {
            let eraseArea = Float.random(in: scale.0...scale.1) * area
            let aspectRatio = Float.random(in: ratio.0...ratio.1)
            
            let eraseH = Int(sqrt(eraseArea * aspectRatio))
            let eraseW = Int(sqrt(eraseArea / aspectRatio))
            
            if eraseH < height && eraseW < width {
                let startH = Int.random(in: 0...(height - eraseH))
                let startW = Int.random(in: 0...(width - eraseW))
                
                for c in 0..<channels {
                    for h in startH..<(startH + eraseH) {
                        for w in startW..<(startW + eraseW) {
                            let idx = c * height * width + h * width + w
                            result.data[idx] = Float.random(in: 0..<1)  // Random fill
                        }
                    }
                }
                break
            }
        }
        
        return result
    }
    
    /// Normalize image
    private func normalize(_ image: Matrix, mean: [Float], std: [Float], channels: Int) -> Matrix {
        var result = image
        let pixelsPerChannel = image.rows / channels
        
        for c in 0..<channels {
            let m = c < mean.count ? mean[c] : mean[0]
            let s = c < std.count ? std[c] : std[0]
            
            for i in 0..<pixelsPerChannel {
                let idx = c * pixelsPerChannel + i
                result.data[idx] = (result.data[idx] - m) / s
            }
        }
        
        return result
    }
}

// MARK: - MixUp Augmentation

/// MixUp data augmentation - mixes two samples and their labels
public struct MixUp {
    /// Alpha parameter for Beta distribution
    public let alpha: Float
    
    public init(alpha: Float = 0.2) {
        self.alpha = alpha
    }
    
    /// Apply MixUp to two samples
    /// - Parameters:
    ///   - image1: First image
    ///   - label1: First label (one-hot encoded)
    ///   - image2: Second image
    ///   - label2: Second label (one-hot encoded)
    /// - Returns: Mixed image and label
    public func apply(image1: Matrix, label1: Matrix, image2: Matrix, label2: Matrix) -> (image: Matrix, label: Matrix) {
        // Sample lambda from Beta(alpha, alpha)
        let lambda = betaSample(alpha: alpha, beta: alpha)
        
        // Mix images
        var mixedImage = Matrix(rows: image1.rows, cols: 1)
        for i in 0..<image1.data.count {
            mixedImage.data[i] = lambda * image1.data[i] + (1 - lambda) * image2.data[i]
        }
        
        // Mix labels
        var mixedLabel = Matrix(rows: label1.rows, cols: 1)
        for i in 0..<label1.data.count {
            mixedLabel.data[i] = lambda * label1.data[i] + (1 - lambda) * label2.data[i]
        }
        
        return (mixedImage, mixedLabel)
    }
    
    /// Simple approximation of Beta distribution sample
    private func betaSample(alpha: Float, beta: Float) -> Float {
        // Using the gamma distribution relationship
        // This is a simple approximation
        let u = Float.random(in: 0..<1)
        let v = Float.random(in: 0..<1)
        
        // Simple symmetric beta approximation when alpha == beta
        return min(max(u / (u + v), 0.0), 1.0)
    }
}

// MARK: - CutMix Augmentation

/// CutMix data augmentation - cuts and pastes regions between images
public struct CutMix {
    /// Alpha parameter for Beta distribution
    public let alpha: Float
    
    public init(alpha: Float = 1.0) {
        self.alpha = alpha
    }
    
    /// Apply CutMix to two samples
    public func apply(image1: Matrix, label1: Matrix, image2: Matrix, label2: Matrix, height: Int, width: Int, channels: Int = 1) -> (image: Matrix, label: Matrix) {
        // Sample lambda from Beta(alpha, alpha)
        let lambda = Float.random(in: 0..<1)  // Simplified
        
        // Compute cut dimensions
        let cutRatio = sqrt(1 - lambda)
        let cutW = Int(Float(width) * cutRatio)
        let cutH = Int(Float(height) * cutRatio)
        
        // Random center
        let cx = Int.random(in: 0..<width)
        let cy = Int.random(in: 0..<height)
        
        let bbx1 = max(0, cx - cutW / 2)
        let bby1 = max(0, cy - cutH / 2)
        let bbx2 = min(width, cx + cutW / 2)
        let bby2 = min(height, cy + cutH / 2)
        
        // Copy image1 and paste region from image2
        var mixedImage = image1
        
        for c in 0..<channels {
            for h in bby1..<bby2 {
                for w in bbx1..<bbx2 {
                    let idx = c * height * width + h * width + w
                    mixedImage.data[idx] = image2.data[idx]
                }
            }
        }
        
        // Compute actual lambda based on cut area
        let actualLambda = 1 - Float((bbx2 - bbx1) * (bby2 - bby1)) / Float(width * height)
        
        // Mix labels
        var mixedLabel = Matrix(rows: label1.rows, cols: 1)
        for i in 0..<label1.data.count {
            mixedLabel.data[i] = actualLambda * label1.data[i] + (1 - actualLambda) * label2.data[i]
        }
        
        return (mixedImage, mixedLabel)
    }
}
