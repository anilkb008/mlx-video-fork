// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MLXVideoApp",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../MLXVideo"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    ],
    targets: [
        .executableTarget(
            name: "MLXVideoApp",
            dependencies: [
                .product(name: "MLXVideo", package: "MLXVideo"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "MLXVideoApp"
        ),
    ]
)
