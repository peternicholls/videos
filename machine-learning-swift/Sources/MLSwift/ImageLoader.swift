/// ImageLoader.swift
/// Native image loading utilities for JPEG and PNG formats
/// Uses Apple's ImageIO framework

import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

// MARK: - Image Loader

/// Utility for loading images from files
public enum ImageLoader {
    
    /// Load an image from a file URL
    /// - Parameter url: URL of the image file
    /// - Returns: Image data as (pixels, width, height, channels)
    /// - Throws: ImageLoaderError if loading fails
    public static func load(from url: URL) throws -> ImageData {
        #if canImport(CoreGraphics)
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw ImageLoaderError.failedToCreateImageSource
        }
        
        guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            throw ImageLoaderError.failedToCreateImage
        }
        
        return try loadFromCGImage(cgImage)
        #else
        throw ImageLoaderError.coreGraphicsNotAvailable
        #endif
    }
    
    /// Load an image from data
    /// - Parameter data: Image data (JPEG, PNG, etc.)
    /// - Returns: Image data as (pixels, width, height, channels)
    /// - Throws: ImageLoaderError if loading fails
    public static func load(from data: Data) throws -> ImageData {
        #if canImport(CoreGraphics)
        guard let imageSource = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw ImageLoaderError.failedToCreateImageSource
        }
        
        guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            throw ImageLoaderError.failedToCreateImage
        }
        
        return try loadFromCGImage(cgImage)
        #else
        throw ImageLoaderError.coreGraphicsNotAvailable
        #endif
    }
    
    #if canImport(CoreGraphics)
    /// Load image data from a CGImage
    private static func loadFromCGImage(_ cgImage: CGImage) throws -> ImageData {
        let width = cgImage.width
        let height = cgImage.height
        let bitsPerComponent = 8
        let channels = 4  // RGBA
        let bytesPerPixel = channels
        let bytesPerRow = width * bytesPerPixel
        
        // Create bitmap context
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw ImageLoaderError.failedToCreateColorSpace
        }
        
        var pixelData = [UInt8](repeating: 0, count: width * height * channels)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ImageLoaderError.failedToCreateContext
        }
        
        // Draw image into context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to float [0, 1] and create Matrix
        // Format: (channels * height * width, 1) for RGB or (height * width, 1) for grayscale
        var floatPixels = [Float](repeating: 0, count: width * height * channels)
        
        for i in 0..<(width * height) {
            let pixelIndex = i * channels
            floatPixels[i] = Float(pixelData[pixelIndex]) / 255.0                      // R
            floatPixels[width * height + i] = Float(pixelData[pixelIndex + 1]) / 255.0  // G
            floatPixels[2 * width * height + i] = Float(pixelData[pixelIndex + 2]) / 255.0  // B
            floatPixels[3 * width * height + i] = Float(pixelData[pixelIndex + 3]) / 255.0  // A
        }
        
        return ImageData(
            pixels: Matrix(rows: channels * height * width, cols: 1, data: floatPixels),
            width: width,
            height: height,
            channels: channels
        )
    }
    #endif
}

// MARK: - Image Data Structure

/// Struct holding loaded image data
public struct ImageData {
    /// Pixel values as Matrix (channels * height * width, 1)
    public let pixels: Matrix
    
    /// Image width
    public let width: Int
    
    /// Image height
    public let height: Int
    
    /// Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
    public let channels: Int
    
    /// Convert to grayscale using luminance formula
    /// Y = 0.299*R + 0.587*G + 0.114*B
    public func toGrayscale() -> ImageData {
        guard channels >= 3 else { return self }
        
        var grayPixels = [Float](repeating: 0, count: width * height)
        
        for i in 0..<(width * height) {
            let r = pixels.data[i]
            let g = pixels.data[width * height + i]
            let b = pixels.data[2 * width * height + i]
            grayPixels[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }
        
        return ImageData(
            pixels: Matrix(rows: height * width, cols: 1, data: grayPixels),
            width: width,
            height: height,
            channels: 1
        )
    }
    
    /// Resize image to new dimensions using bilinear interpolation
    /// - Parameters:
    ///   - newWidth: Target width
    ///   - newHeight: Target height
    /// - Returns: Resized image data
    public func resize(to newWidth: Int, newHeight: Int) -> ImageData {
        var resizedPixels = [Float](repeating: 0, count: channels * newHeight * newWidth)
        
        let scaleX = Float(width) / Float(newWidth)
        let scaleY = Float(height) / Float(newHeight)
        
        for c in 0..<channels {
            for y in 0..<newHeight {
                for x in 0..<newWidth {
                    let srcX = Float(x) * scaleX
                    let srcY = Float(y) * scaleY
                    
                    let value = bilinearInterpolate(channel: c, x: srcX, y: srcY)
                    
                    let dstIdx = c * newHeight * newWidth + y * newWidth + x
                    resizedPixels[dstIdx] = value
                }
            }
        }
        
        return ImageData(
            pixels: Matrix(rows: channels * newHeight * newWidth, cols: 1, data: resizedPixels),
            width: newWidth,
            height: newHeight,
            channels: channels
        )
    }
    
    /// Bilinear interpolation helper
    private func bilinearInterpolate(channel: Int, x: Float, y: Float) -> Float {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        
        let dx = x - Float(x0)
        let dy = y - Float(y0)
        
        let baseIdx = channel * height * width
        
        let v00 = pixels.data[baseIdx + y0 * width + x0]
        let v01 = pixels.data[baseIdx + y0 * width + x1]
        let v10 = pixels.data[baseIdx + y1 * width + x0]
        let v11 = pixels.data[baseIdx + y1 * width + x1]
        
        return (1 - dy) * (1 - dx) * v00 +
               (1 - dy) * dx * v01 +
               dy * (1 - dx) * v10 +
               dy * dx * v11
    }
    
    /// Convert to Matrix for neural network input
    /// - Returns: Flattened pixel matrix
    public func toMatrix() -> Matrix {
        return pixels
    }
    
    /// Extract a specific channel
    /// - Parameter channel: Channel index (0 = R, 1 = G, 2 = B for RGB)
    /// - Returns: Single-channel image data
    public func extractChannel(_ channel: Int) -> ImageData {
        precondition(channel >= 0 && channel < channels, "Channel index out of bounds")
        
        var channelPixels = [Float](repeating: 0, count: width * height)
        let baseIdx = channel * height * width
        
        for i in 0..<(width * height) {
            channelPixels[i] = pixels.data[baseIdx + i]
        }
        
        return ImageData(
            pixels: Matrix(rows: height * width, cols: 1, data: channelPixels),
            width: width,
            height: height,
            channels: 1
        )
    }
    
    /// Normalize pixel values to [0, 1]
    /// - Returns: Normalized image data
    public func normalize() -> ImageData {
        var normalized = pixels
        
        if let maxVal = normalized.data.max(), let minVal = normalized.data.min() {
            let range = maxVal - minVal
            if range > 0 {
                for i in 0..<normalized.data.count {
                    normalized.data[i] = (normalized.data[i] - minVal) / range
                }
            }
        }
        
        return ImageData(
            pixels: normalized,
            width: width,
            height: height,
            channels: channels
        )
    }
    
    /// Standardize pixel values (zero mean, unit variance)
    /// - Returns: Standardized image data
    public func standardize() -> ImageData {
        let count = Float(pixels.data.count)
        let mean = pixels.data.reduce(0, +) / count
        
        let variance = pixels.data.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / count
        let stdDev = sqrt(variance)
        
        var standardized = pixels
        if stdDev > 0 {
            for i in 0..<standardized.data.count {
                standardized.data[i] = (standardized.data[i] - mean) / stdDev
            }
        }
        
        return ImageData(
            pixels: standardized,
            width: width,
            height: height,
            channels: channels
        )
    }
    
    /// Center crop the image
    /// - Parameters:
    ///   - cropWidth: Width of the crop
    ///   - cropHeight: Height of the crop
    /// - Returns: Cropped image data
    public func centerCrop(width cropWidth: Int, height cropHeight: Int) -> ImageData {
        precondition(cropWidth <= width && cropHeight <= height, "Crop size must be smaller than image")
        
        let startX = (width - cropWidth) / 2
        let startY = (height - cropHeight) / 2
        
        var croppedPixels = [Float](repeating: 0, count: channels * cropHeight * cropWidth)
        
        for c in 0..<channels {
            for y in 0..<cropHeight {
                for x in 0..<cropWidth {
                    let srcIdx = c * height * width + (startY + y) * width + (startX + x)
                    let dstIdx = c * cropHeight * cropWidth + y * cropWidth + x
                    croppedPixels[dstIdx] = pixels.data[srcIdx]
                }
            }
        }
        
        return ImageData(
            pixels: Matrix(rows: channels * cropHeight * cropWidth, cols: 1, data: croppedPixels),
            width: cropWidth,
            height: cropHeight,
            channels: channels
        )
    }
}

// MARK: - Image Loader Errors

public enum ImageLoaderError: Error, LocalizedError {
    case failedToCreateImageSource
    case failedToCreateImage
    case failedToCreateColorSpace
    case failedToCreateContext
    case coreGraphicsNotAvailable
    case unsupportedFormat
    
    public var errorDescription: String? {
        switch self {
        case .failedToCreateImageSource:
            return "Failed to create image source from file"
        case .failedToCreateImage:
            return "Failed to create image from source"
        case .failedToCreateColorSpace:
            return "Failed to create color space"
        case .failedToCreateContext:
            return "Failed to create bitmap context"
        case .coreGraphicsNotAvailable:
            return "CoreGraphics is not available on this platform"
        case .unsupportedFormat:
            return "Unsupported image format"
        }
    }
}

// MARK: - Batch Image Loading

extension ImageLoader {
    
    /// Load multiple images from a directory
    /// - Parameters:
    ///   - directory: Directory URL
    ///   - extensions: File extensions to load (default: ["jpg", "jpeg", "png"])
    /// - Returns: Array of loaded images
    public static func loadDirectory(
        at directory: URL,
        extensions: [String] = ["jpg", "jpeg", "png"]
    ) throws -> [ImageData] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        
        var images: [ImageData] = []
        
        for url in contents {
            let ext = url.pathExtension.lowercased()
            if extensions.contains(ext) {
                do {
                    let image = try load(from: url)
                    images.append(image)
                } catch {
                    print("Warning: Failed to load \(url.lastPathComponent): \(error)")
                }
            }
        }
        
        return images
    }
    
    /// Load images and resize to uniform dimensions
    /// - Parameters:
    ///   - directory: Directory URL
    ///   - targetWidth: Target width for all images
    ///   - targetHeight: Target height for all images
    ///   - grayscale: Convert to grayscale
    /// - Returns: Array of uniformly-sized images
    public static func loadAndPreprocess(
        from directory: URL,
        targetWidth: Int,
        targetHeight: Int,
        grayscale: Bool = false
    ) throws -> [ImageData] {
        let images = try loadDirectory(at: directory)
        
        return images.map { image in
            var processed = image.resize(to: targetWidth, newHeight: targetHeight)
            if grayscale {
                processed = processed.toGrayscale()
            }
            return processed.normalize()
        }
    }
}
