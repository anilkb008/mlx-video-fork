// ModelConfig.swift - Base configuration types for MLX Video models

import Foundation
import MLX

// MARK: - Base Configuration Protocol

public protocol ModelConfig: Codable, Sendable {
    var modelType: String { get }
}

// MARK: - Weight Loading

public enum WeightFormat {
    case safetensors
}

public struct QuantizationConfig: Codable, Sendable {
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}
