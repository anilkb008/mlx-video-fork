// LTXTransformer.swift
// Transformer block for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/transformer.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Modality

/// Input modality data (video or audio).
public struct Modality: Sendable {
    public let latent: MLXArray
    public let timesteps: MLXArray
    public let positions: MLXArray
    public let context: MLXArray
    public let enabled: Bool
    public let contextMask: MLXArray?
    /// Optional precomputed positional embeddings (RoPE) to avoid recomputation.
    public let positionalEmbeddings: (MLXArray, MLXArray)?
    /// Raw sigma value (scalar per batch) for prompt adaln (LTX-2.3).
    public let sigma: MLXArray?

    public init(
        latent: MLXArray,
        timesteps: MLXArray,
        positions: MLXArray,
        context: MLXArray,
        enabled: Bool = true,
        contextMask: MLXArray? = nil,
        positionalEmbeddings: (MLXArray, MLXArray)? = nil,
        sigma: MLXArray? = nil
    ) {
        self.latent = latent
        self.timesteps = timesteps
        self.positions = positions
        self.context = context
        self.enabled = enabled
        self.contextMask = contextMask
        self.positionalEmbeddings = positionalEmbeddings
        self.sigma = sigma
    }
}

// MARK: - TransformerArgs

/// Preprocessed arguments for a transformer block.
public struct TransformerArgs {
    public var x: MLXArray
    public let context: MLXArray
    public let contextMask: MLXArray?
    public let timesteps: MLXArray
    public let embeddedTimestep: MLXArray
    public let positionalEmbeddings: (MLXArray, MLXArray)
    public let crossPositionalEmbeddings: (MLXArray, MLXArray)?
    public let crossScaleShiftTimestep: MLXArray?
    public let crossGateTimestep: MLXArray?
    public let enabled: Bool
    /// LTX-2.3: prompt-conditioned timestep embeddings for cross-attention.
    public let promptTimesteps: MLXArray?
    public let promptEmbeddedTimestep: MLXArray?

    public init(
        x: MLXArray,
        context: MLXArray,
        contextMask: MLXArray?,
        timesteps: MLXArray,
        embeddedTimestep: MLXArray,
        positionalEmbeddings: (MLXArray, MLXArray),
        crossPositionalEmbeddings: (MLXArray, MLXArray)?,
        crossScaleShiftTimestep: MLXArray?,
        crossGateTimestep: MLXArray?,
        enabled: Bool,
        promptTimesteps: MLXArray? = nil,
        promptEmbeddedTimestep: MLXArray? = nil
    ) {
        self.x = x
        self.context = context
        self.contextMask = contextMask
        self.timesteps = timesteps
        self.embeddedTimestep = embeddedTimestep
        self.positionalEmbeddings = positionalEmbeddings
        self.crossPositionalEmbeddings = crossPositionalEmbeddings
        self.crossScaleShiftTimestep = crossScaleShiftTimestep
        self.crossGateTimestep = crossGateTimestep
        self.enabled = enabled
        self.promptTimesteps = promptTimesteps
        self.promptEmbeddedTimestep = promptEmbeddedTimestep
    }

    /// Return a copy with updated x.
    public func withX(_ newX: MLXArray) -> TransformerArgs {
        TransformerArgs(
            x: newX,
            context: context,
            contextMask: contextMask,
            timesteps: timesteps,
            embeddedTimestep: embeddedTimestep,
            positionalEmbeddings: positionalEmbeddings,
            crossPositionalEmbeddings: crossPositionalEmbeddings,
            crossScaleShiftTimestep: crossScaleShiftTimestep,
            crossGateTimestep: crossGateTimestep,
            enabled: enabled,
            promptTimesteps: promptTimesteps,
            promptEmbeddedTimestep: promptEmbeddedTimestep
        )
    }
}

// MARK: - RMS Norm helper

/// Affine-free RMS normalization using MLXFast.
func rmsNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let weight = MLXArray.ones([x.shape[x.ndim - 1]], type: x.dtype)
    return MLXFast.rmsNorm(x, weight: weight, eps: eps)
}

// MARK: - BasicAVTransformerBlock

/// Audio-Video transformer block with cross-modal attention.
/// Supports video-only, audio-only, or combined audio-video processing.
public class BasicAVTransformerBlock: Module {

    let idx: Int
    let normEps: Float
    let hasPromptAdaln: Bool

    // Video components
    var attn1: LTXAttention?
    var attn2: LTXAttention?
    var ff: LTXFeedForward?
    var scaleShiftTable: MLXArray?
    var promptScaleShiftTable: MLXArray?

    // Audio components (skipped for video-only, kept as placeholders)
    var audioAttn1: LTXAttention?
    var audioAttn2: LTXAttention?
    var audioFf: LTXFeedForward?
    var audioScaleShiftTable: MLXArray?
    var audioPromptScaleShiftTable: MLXArray?

    public init(
        idx: Int,
        video: TransformerConfig? = nil,
        audio: TransformerConfig? = nil,
        ropeType: LTXRopeType = .interleaved,
        normEps: Float = 1e-6,
        hasPromptAdaln: Bool = false
    ) {
        self.idx = idx
        self.normEps = normEps
        self.hasPromptAdaln = hasPromptAdaln

        // Video components
        if let video = video {
            self.attn1 = LTXAttention(
                queryDim: video.dim,
                heads: video.heads,
                dimHead: video.dHead,
                contextDim: nil,
                ropeType: ropeType,
                normEps: normEps,
                hasGateLogits: hasPromptAdaln
            )
            self.attn2 = LTXAttention(
                queryDim: video.dim,
                contextDim: video.contextDim,
                heads: video.heads,
                dimHead: video.dHead,
                ropeType: ropeType,
                normEps: normEps,
                hasGateLogits: hasPromptAdaln
            )
            self.ff = LTXFeedForward(dim: video.dim, dimOut: video.dim)
            let numAdaParams = hasPromptAdaln ? 9 : 6
            self.scaleShiftTable = MLXArray.zeros([numAdaParams, video.dim])

            if hasPromptAdaln {
                self.promptScaleShiftTable = MLXArray.zeros([2, video.dim])
            }
        }

        // Audio components
        if let audio = audio {
            self.audioAttn1 = LTXAttention(
                queryDim: audio.dim,
                heads: audio.heads,
                dimHead: audio.dHead,
                contextDim: nil,
                ropeType: ropeType,
                normEps: normEps,
                hasGateLogits: hasPromptAdaln
            )
            self.audioAttn2 = LTXAttention(
                queryDim: audio.dim,
                contextDim: audio.contextDim,
                heads: audio.heads,
                dimHead: audio.dHead,
                ropeType: ropeType,
                normEps: normEps,
                hasGateLogits: hasPromptAdaln
            )
            self.audioFf = LTXFeedForward(dim: audio.dim, dimOut: audio.dim)
            let numAudioAdaParams = hasPromptAdaln ? 9 : 6
            self.audioScaleShiftTable = MLXArray.zeros([numAudioAdaParams, audio.dim])

            if hasPromptAdaln {
                self.audioPromptScaleShiftTable = MLXArray.zeros([2, audio.dim])
            }
        }
    }

    /// Get adaptive normalization values from scale-shift table.
    func getAdaValues(
        scaleShiftTable: MLXArray,
        batchSize: Int,
        timestep: MLXArray,
        startIdx: Int,
        endIdx: Int
    ) -> [MLXArray] {
        let numAdaParams = scaleShiftTable.shape[0]

        // Table slice: (numSelected, dim)
        let tableSlice = scaleShiftTable[startIdx..<endIdx]
        // Add batch and sequence dimensions: (1, 1, numSelected, dim)
        let tableExpanded = tableSlice.expandedDimensions(axes: [0, 0])

        // timestep: (B, seq, numParams * dim) -> reshape to (B, seq, numParams, dim)
        let timestepReshaped = timestep.reshaped([batchSize, timestep.shape[1], numAdaParams, -1])

        // Extract the relevant indices
        let timestepSlice = timestepReshaped[0..., 0..., startIdx..<endIdx, 0...]

        // Add table values to timestep
        let adaValues = tableExpanded + timestepSlice

        // Unbind along the parameter dimension
        let numSliced = adaValues.shape[2]
        var result: [MLXArray] = []
        for i in 0..<numSliced {
            result.append(adaValues[0..., 0..., i, 0...])
        }

        return result
    }

    /// Forward pass through transformer block.
    public func callAsFunction(
        video: TransformerArgs?,
        audio: TransformerArgs?,
        skipVideoSelfAttn: Bool = false,
        skipAudioSelfAttn: Bool = false,
        skipCrossModal: Bool = false
    ) -> (TransformerArgs?, TransformerArgs?) {
        let batchSize = video?.x.shape[0] ?? audio!.x.shape[0]

        var vx = video?.x
        var ax = audio?.x

        let runVx = video != nil && video!.enabled && vx!.size > 0
        let runAx = audio != nil && audio!.enabled && ax!.size > 0

        // Process video self-attention and cross-attention with text
        if runVx, let video = video, let table = scaleShiftTable,
           let attn1 = attn1, let attn2 = attn2, let ff = ff {

            let adaSelfAttn = getAdaValues(
                scaleShiftTable: table, batchSize: vx!.shape[0],
                timestep: video.timesteps, startIdx: 0, endIdx: 3
            )
            let vshiftMsa = adaSelfAttn[0]
            let vscaleMsa = adaSelfAttn[1]
            let vgateMsa = adaSelfAttn[2]

            // Self-attention with RoPE
            let normVx = rmsNorm(vx!, eps: normEps) * (1 + vscaleMsa) + vshiftMsa
            vx = vx! + attn1(
                normVx,
                pe: video.positionalEmbeddings,
                skipAttention: skipVideoSelfAttn
            ) * vgateMsa

            // Cross-attention with text context
            if hasPromptAdaln, let promptTable = promptScaleShiftTable {
                // LTX-2.3: Q modulated by timestep (indices 6-8), context modulated by prompt_adaln
                let adaQ = getAdaValues(
                    scaleShiftTable: table, batchSize: vx!.shape[0],
                    timestep: video.timesteps, startIdx: 6, endIdx: 9
                )
                let vshiftQ = adaQ[0]
                let vscaleQ = adaQ[1]
                let vgateQ = adaQ[2]

                let promptAda = getAdaValues(
                    scaleShiftTable: promptTable, batchSize: vx!.shape[0],
                    timestep: video.promptTimesteps!, startIdx: 0, endIdx: 2
                )
                let vpromptShiftKv = promptAda[0]
                let vpromptScaleKv = promptAda[1]

                let attnInput = rmsNorm(vx!, eps: normEps) * (1 + vscaleQ) + vshiftQ
                let encoderHiddenStates = video.context * (1 + vpromptScaleKv) + vpromptShiftKv
                vx = vx! + attn2(
                    attnInput,
                    context: encoderHiddenStates,
                    mask: video.contextMask
                ) * vgateQ
            } else {
                vx = vx! + attn2(
                    rmsNorm(vx!, eps: normEps),
                    context: video.context,
                    mask: video.contextMask
                )
            }

            // Feed-forward
            let adaFf = getAdaValues(
                scaleShiftTable: table, batchSize: vx!.shape[0],
                timestep: video.timesteps, startIdx: 3, endIdx: 6
            )
            let vshiftMlp = adaFf[0]
            let vscaleMlp = adaFf[1]
            let vgateMlp = adaFf[2]

            let vxScaled = rmsNorm(vx!, eps: normEps) * (1 + vscaleMlp) + vshiftMlp
            vx = vx! + ff(vxScaled) * vgateMlp
        }

        // Process audio self-attention and cross-attention with text
        if runAx, let audio = audio, let table = audioScaleShiftTable,
           let attn1 = audioAttn1, let attn2 = audioAttn2, let ff = audioFf {

            let adaSelfAttn = getAdaValues(
                scaleShiftTable: table, batchSize: ax!.shape[0],
                timestep: audio.timesteps, startIdx: 0, endIdx: 3
            )
            let ashiftMsa = adaSelfAttn[0]
            let ascaleMsa = adaSelfAttn[1]
            let agateMsa = adaSelfAttn[2]

            let normAx = rmsNorm(ax!, eps: normEps) * (1 + ascaleMsa) + ashiftMsa
            ax = ax! + attn1(
                normAx,
                pe: audio.positionalEmbeddings,
                skipAttention: skipAudioSelfAttn
            ) * agateMsa

            if hasPromptAdaln, let promptTable = audioPromptScaleShiftTable {
                let adaQ = getAdaValues(
                    scaleShiftTable: table, batchSize: ax!.shape[0],
                    timestep: audio.timesteps, startIdx: 6, endIdx: 9
                )
                let ashiftQ = adaQ[0]
                let ascaleQ = adaQ[1]
                let agateQ = adaQ[2]

                let promptAda = getAdaValues(
                    scaleShiftTable: promptTable, batchSize: ax!.shape[0],
                    timestep: audio.promptTimesteps!, startIdx: 0, endIdx: 2
                )
                let apromptShiftKv = promptAda[0]
                let apromptScaleKv = promptAda[1]

                let attnInputA = rmsNorm(ax!, eps: normEps) * (1 + ascaleQ) + ashiftQ
                let encoderHiddenStatesA = audio.context * (1 + apromptScaleKv) + apromptShiftKv
                ax = ax! + attn2(
                    attnInputA,
                    context: encoderHiddenStatesA,
                    mask: audio.contextMask
                ) * agateQ
            } else {
                ax = ax! + attn2(
                    rmsNorm(ax!, eps: normEps),
                    context: audio.context,
                    mask: audio.contextMask
                )
            }

            let adaFf = getAdaValues(
                scaleShiftTable: table, batchSize: ax!.shape[0],
                timestep: audio.timesteps, startIdx: 3, endIdx: 6
            )
            let ashiftMlp = adaFf[0]
            let ascaleMlp = adaFf[1]
            let agateMlp = adaFf[2]

            let axScaled = rmsNorm(ax!, eps: normEps) * (1 + ascaleMlp) + ashiftMlp
            ax = ax! + ff(axScaled) * agateMlp
        }

        // Return updated TransformerArgs
        let videoOut = video.map { $0.withX(vx!) }
        let audioOut = audio.map { $0.withX(ax!) }

        return (videoOut, audioOut)
    }
}
