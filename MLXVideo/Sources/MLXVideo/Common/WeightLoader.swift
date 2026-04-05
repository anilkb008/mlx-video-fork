// WeightLoader.swift - Weight loading utilities for MLX Video models

import Foundation
import MLX
import MLXNN

// MARK: - HuggingFace Hub Download

/// Resolve a model path: if it's a local directory use it directly,
/// otherwise download from HuggingFace Hub.
///
/// Supports repo IDs like `dgrauet/ltx-2.3-mlx-q4`.
///
/// - Parameter modelRepo: Local path or HuggingFace repo ID
/// - Returns: Local filesystem path to the model directory
public func getModelPath(_ modelRepo: String) async throws -> String {
    // Check if it's a local path first
    let fm = FileManager.default
    if fm.fileExists(atPath: modelRepo) {
        return modelRepo
    }

    // Try HuggingFace Hub download
    let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-video-models")
    try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    // Use swift-transformers Hub API if available, otherwise manual download
    let repoSlug = modelRepo.replacingOccurrences(of: "/", with: "--")
    let localDir = cacheDir.appendingPathComponent(repoSlug)

    if fm.fileExists(atPath: localDir.path) {
        // Check if config.json exists (basic validation)
        let configPath = localDir.appendingPathComponent("config.json")
        if fm.fileExists(atPath: configPath.path) {
            return localDir.path
        }
    }

    // Download via huggingface-cli or direct API
    print("Downloading model from HuggingFace: \(modelRepo)...")
    let hubURL = "https://huggingface.co/\(modelRepo)/resolve/main/"

    // Fetch file list from the repo
    let apiURL = URL(string: "https://huggingface.co/api/models/\(modelRepo)")!
    let (apiData, _) = try await URLSession.shared.data(from: apiURL)

    struct HFModelInfo: Codable {
        let siblings: [HFSibling]?
    }
    struct HFSibling: Codable {
        let rfilename: String
    }

    let modelInfo = try JSONDecoder().decode(HFModelInfo.self, from: apiData)
    let filesToDownload = (modelInfo.siblings ?? [])
        .map(\.rfilename)
        .filter { $0.hasSuffix(".safetensors") || $0.hasSuffix(".json") }

    try fm.createDirectory(at: localDir, withIntermediateDirectories: true)

    for filename in filesToDownload {
        let fileURL = URL(string: hubURL + filename)!
        let destPath = localDir.appendingPathComponent(filename)

        // Create subdirectories if needed
        let parentDir = destPath.deletingLastPathComponent()
        try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)

        if fm.fileExists(atPath: destPath.path) {
            continue // Skip already downloaded files
        }

        print("  Downloading \(filename)...")
        let (data, _) = try await URLSession.shared.data(from: fileURL)
        try data.write(to: destPath)
    }

    print("Model downloaded to: \(localDir.path)")
    return localDir.path
}

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

/// Load all safetensors files from a directory into a single dictionary.
public func loadAllWeights(from directory: String) throws -> [String: MLXArray] {
    var allWeights: [String: MLXArray] = [:]
    for path in findWeightFiles(in: directory) {
        let fileWeights = try loadWeightsDict(from: path)
        for (k, v) in fileWeights {
            allWeights[k] = v
        }
    }
    return allWeights
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

/// Load raw config.json as a dictionary (for reading quantization settings etc).
public func loadRawConfig(from directory: String) throws -> [String: Any] {
    let configPath = (directory as NSString).appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: configPath) else {
        throw WeightLoadError.configNotFound(configPath)
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
    guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw WeightLoadError.invalidFormat("config.json is not a JSON object")
    }
    return dict
}

// MARK: - Quantization

/// Quantization configuration read from config.json.
public struct QuantizationConfig: Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    public init(bits: Int, groupSize: Int, mode: String = "affine") {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }
}

/// Read quantization config from a model's config.json (if present).
public func readQuantizationConfig(from directory: String) throws -> QuantizationConfig? {
    let rawConfig = try loadRawConfig(from: directory)
    guard let quantDict = rawConfig["quantization"] as? [String: Any] else {
        return nil
    }
    guard let bits = quantDict["bits"] as? Int,
          let groupSize = quantDict["group_size"] as? Int else {
        return nil
    }
    let mode = quantDict["mode"] as? String ?? "affine"
    return QuantizationConfig(bits: bits, groupSize: groupSize, mode: mode)
}

/// Apply quantization to a model: replaces Linear layers with QuantizedLinear stubs.
///
/// Must be called BEFORE loading weights so the model structure matches the weight keys
/// (quantized weights have `scales`, `biases`, and packed `weight` instead of a plain `weight`).
///
/// - Parameters:
///   - model: The model to quantize
///   - config: Quantization configuration (bits, groupSize)
///   - weights: Weight dictionary (used to detect which layers are quantized via `.scales` keys)
public func applyQuantization(
    model: Module,
    config: QuantizationConfig,
    weights: [String: MLXArray]
) {
    // Determine which layers have quantized weights (have .scales keys)
    let quantizedPrefixes = Set(
        weights.keys
            .filter { $0.hasSuffix(".scales") }
            .map { String($0.dropLast(".scales".count)) }
    )

    QuantizedLinear.quantize(
        model: model,
        groupSize: config.groupSize,
        bits: config.bits,
        predicate: { path, module in
            guard module is Linear else { return false }
            // Only quantize layers that have .scales in the weights
            return quantizedPrefixes.contains(path)
        }
    )
}

// MARK: - Errors

public enum WeightLoadError: LocalizedError {
    case fileNotFound(String)
    case configNotFound(String)
    case invalidFormat(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Weight file not found: \(path)"
        case .configNotFound(let path):
            return "Config file not found: \(path)"
        case .invalidFormat(let msg):
            return "Invalid format: \(msg)"
        case .downloadFailed(let msg):
            return "Download failed: \(msg)"
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
