// WeightLoader.swift - Weight loading utilities for MLX Video models

import Foundation
import MLX
import MLXNN

// MARK: - Weight Loading

/// Load safetensors weights from a file path.
public func loadWeights(from path: String) throws -> [(String, MLXArray)] {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path) else {
        throw WeightLoadError.fileNotFound(path)
    }
    let weights = try MLX.loadArrays(url: url)
    return weights.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
}

/// Load safetensors weights as a dictionary.
public func loadWeightsDict(from path: String) throws -> [String: MLXArray] {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path) else {
        throw WeightLoadError.fileNotFound(path)
    }
    return try MLX.loadArrays(url: url)
}

/// Load and cast weights to a specific dtype.
public func loadWeights(from path: String, dtype: DType) throws -> [(String, MLXArray)] {
    let weights = try loadWeights(from: path)
    return weights.map { (key, value) in
        if value.dtype == .float32 && dtype != .float32 {
            return (key, value.asType(dtype))
        }
        return (key, value)
    }
}

/// Load config.json from a model directory.
public func loadModelConfig<T: Decodable>(from directory: String, type: T.Type) throws -> T {
    let configPath = (directory as NSString).appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: configPath) else {
        throw WeightLoadError.configNotFound(configPath)
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
    return try JSONDecoder().decode(type, from: data)
}

// MARK: - Errors

public enum WeightLoadError: LocalizedError {
    case fileNotFound(String)
    case configNotFound(String)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Weight file not found: \(path)"
        case .configNotFound(let path):
            return "Config file not found: \(path)"
        case .invalidFormat(let msg):
            return "Invalid format: \(msg)"
        }
    }
}

// MARK: - Model Directory Utilities

/// Discover weight files in a model directory.
public func findWeightFiles(in directory: String) -> [String] {
    let fm = FileManager.default
    guard let contents = try? fm.contentsOfDirectory(atPath: directory) else {
        return []
    }
    return contents
        .filter { $0.hasSuffix(".safetensors") }
        .map { (directory as NSString).appendingPathComponent($0) }
        .sorted()
}
