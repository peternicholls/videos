// swift-tools-version: 5.9
// Package definition for MLSwift - A Swift/Metal neural network library
// Optimized for macOS with Apple Silicon (M1/M2/M3+)

import PackageDescription

let package = Package(
    name: "MLSwift",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "MLSwift",
            targets: ["MLSwift"]),
        .executable(
            name: "MLSwiftExample",
            targets: ["MLSwiftExample"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MLSwift",
            dependencies: [],
            path: "Sources/MLSwift"),
        .executableTarget(
            name: "MLSwiftExample",
            dependencies: ["MLSwift"],
            path: "Sources/MLSwiftExample"),
        .testTarget(
            name: "MLSwiftTests",
            dependencies: ["MLSwift"],
            path: "Tests/MLSwiftTests")
    ]
)
