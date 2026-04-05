// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MLXVideo",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MLXVideo", targets: ["MLXVideo"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "MLXVideo",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
    ]
)
