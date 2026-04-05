// WanVAE.swift - 3D VAE for Wan2 (compression 4x8x8)
// Ported from mlx_video/models/wan_2/vae.py

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// Temporal cache depth for causal convolution.
private let cacheT = 2

// MARK: - Per-channel normalization statistics for z_dim=16

public let vaeMean: [Float] = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
]

public let vaeStd: [Float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

// MARK: - CausalConv3d

/// 3D convolution with causal temporal padding.
public class CausalConv3d: Module {
    let kernelSize: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let causalPadT: Int
    let padH: Int
    let padW: Int

    /// Weight shape: [O, D, H, W, I] (MLX Conv3d layout)
    public var weight: MLXArray
    public var bias: MLXArray

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: IntOrTriple,
        stride: IntOrTriple = .int(1),
        padding: IntOrTriple = .int(0)
    ) {
        let ks = kernelSize.triple
        let st = stride.triple
        let pd = padding.triple

        self.kernelSize = ks
        self.stride = st
        // Causal padding: k - stride (pads left only, no future context)
        self.causalPadT = ks.0 - st.0
        self.padH = pd.1
        self.padW = pd.2

        self.weight = MLXArray.zeros([outChannels, ks.0, ks.1, ks.2, inChannels])
        self.bias = MLXArray.zeros([outChannels])
    }

    public func callAsFunction(_ x: MLXArray, cacheX: MLXArray? = nil) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4)
        var xVar = x
        var causalPad = causalPadT

        if let cache = cacheX, causalPad > 0 {
            xVar = concatenated([cache, xVar], axis: 2)
            causalPad = max(0, causalPad - cache.dim(2))
        }

        if causalPad > 0 {
            let padT = MLXArray.zeros([b, c, causalPad, h, w]).asType(xVar.dtype)
            xVar = concatenated([padT, xVar], axis: 2)
        }

        if padH > 0 || padW > 0 {
            xVar = padded(xVar, widths: [
                .init(0, 0), .init(0, 0), .init(0, 0),
                .init(padH, padH), .init(padW, padW),
            ])
        }

        xVar = xVar.transposed(0, 2, 3, 4, 1)  // [B, T, H, W, C]
        let out = conv3d(xVar)
        return out.transposed(0, 4, 1, 2, 3)  // [B, O, T', H', W']
    }

    /// 3D conv via sliding window + 2D conv per time step.
    private func conv3d(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3), cIn = x.dim(4)
        let kt = kernelSize.0, kh = kernelSize.1, kw = kernelSize.2
        let st = stride.0, sh = stride.1, sw = stride.2
        let tOut = (t - kt) / st + 1

        // Pre-reshape weight: [O, D, H, W, I] -> [O, H, W, D*I]
        let w2d = weight.transposed(0, 2, 3, 1, 4).reshaped(
            weight.dim(0), kh, kw, kt * cIn
        )

        var outputs = [MLXArray]()
        for tI in 0..<tOut {
            let tStart = tI * st
            let window = x[0..., tStart..<(tStart + kt)]
            let windowReshaped = window.transposed(0, 2, 3, 1, 4).reshaped(b, h, w, kt * cIn)
            let out2d = conv2d(windowReshaped, w2d, stride: [sh, sw]) + bias
            outputs.append(out2d)
        }
        return stacked(outputs, axis: 1)
    }
}

/// Helper for int-or-triple parameters.
public enum IntOrTriple {
    case int(Int)
    case triple(Int, Int, Int)

    var triple: (Int, Int, Int) {
        switch self {
        case .int(let v): return (v, v, v)
        case .triple(let a, let b, let c): return (a, b, c)
        }
    }
}

// MARK: - RMS_norm (VAE)

/// Channel-first L2 normalization matching original Wan VAE.
public class VaeRMSNorm: Module, UnaryLayer {
    let channelFirst: Bool
    let scale: Float
    public var gamma: MLXArray

    public init(dim: Int, channelFirst: Bool = true, images: Bool = true) {
        self.channelFirst = channelFirst
        self.scale = Float(Foundation.sqrt(Double(dim)))
        if channelFirst {
            if images {
                self.gamma = MLXArray.ones([dim, 1, 1])
            } else {
                self.gamma = MLXArray.ones([dim, 1, 1, 1])
            }
        } else {
            self.gamma = MLXArray.ones([dim])
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normDim = channelFirst ? 1 : -1
        // L2 normalize along channel dim (matches F.normalize)
        let norm = sqrt(clip(sum(x * x, axis: normDim, keepDims: true), min: 1e-12))
        return (x / norm) * scale * gamma
    }
}

// MARK: - ResidualBlock

/// Residual block with causal 3D convolutions.
///
/// Uses list-based storage to match original PyTorch nn.Sequential key hierarchy:
/// residual[0]=norm, [1]=SiLU(no params), [2]=conv, [3]=norm, [4]=SiLU, [5]=Dropout, [6]=conv.
/// Weight keys: residual.0.gamma, residual.2.weight, residual.3.gamma, residual.6.weight, etc.
public class ResidualBlock: Module {
    /// Norm at index 0
    let norm0: VaeRMSNorm
    /// Conv at index 2
    let conv2: CausalConv3d
    /// Norm at index 3
    let norm3: VaeRMSNorm
    /// Conv at index 6
    let conv6: CausalConv3d
    let shortcut: CausalConv3d?

    public init(inDim: Int, outDim: Int) {
        self.norm0 = VaeRMSNorm(dim: inDim, channelFirst: true, images: false)
        self.conv2 = CausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: .int(3), padding: .int(1))
        self.norm3 = VaeRMSNorm(dim: outDim, channelFirst: true, images: false)
        self.conv6 = CausalConv3d(inChannels: outDim, outChannels: outDim, kernelSize: .int(3), padding: .int(1))
        self.shortcut = inDim != outDim
            ? CausalConv3d(inChannels: inDim, outChannels: outDim, kernelSize: .int(1))
            : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = shortcut != nil ? shortcut!(x) : x
        var xVar = silu(norm0(x))
        xVar = conv2(xVar)
        xVar = silu(norm3(xVar))
        xVar = conv6(xVar)
        return xVar + h
    }
}

// MARK: - AttentionBlock (VAE)

/// Single-head spatial self-attention for VAE.
public class VaeAttentionBlock: Module {
    @ModuleInfo public var norm: VaeRMSNorm
    @ModuleInfo public var toQkv: Conv2d
    @ModuleInfo public var proj: Conv2d

    public init(dim: Int) {
        self._norm.wrappedValue = VaeRMSNorm(dim: dim, channelFirst: true, images: true)
        self._toQkv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4)

        // [B,C,T,H,W] -> [B,T,C,H,W] -> [BT,C,H,W] -> norm -> [BT,H,W,C]
        var xVar = x.transposed(0, 2, 1, 3, 4).reshaped(b * t, c, h, w)
        xVar = norm(xVar)
        xVar = xVar.transposed(0, 2, 3, 1)  // [BT, H, W, C]

        let qkv = toQkv(xVar)  // [BT, H, W, 3C]
        let qkvR = qkv.reshaped(b * t, h * w, 3, c).transposed(2, 0, 1, 3)
        let q = qkvR[0].expandedDimensions(axis: 1)  // [BT, 1, HW, C]
        let k = qkvR[1].expandedDimensions(axis: 1)
        let v = qkvR[2].expandedDimensions(axis: 1)

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(c), -0.5)
        )
        out = out.squeezed(axis: 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]
        out = proj(out)  // [BT, H, W, C]
        out = out.reshaped(b, t, h, w, c).transposed(0, 4, 1, 2, 3)  // [B, C, T, H, W]
        return out + identity
    }
}

// MARK: - Resample

/// Resample block supporting upsample and downsample modes.
public class Resample: Module {
    let mode: String
    let dim: Int

    @ModuleInfo public var resample1: Conv2d  // resample[1]
    @ModuleInfo public var timeConv: CausalConv3d?

    public init(dim: Int, mode: String) {
        precondition(["upsample2d", "upsample3d", "downsample2d", "downsample3d"].contains(mode))
        self.mode = mode
        self.dim = dim

        if mode.hasPrefix("upsample") {
            self._resample1.wrappedValue = Conv2d(
                inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1
            )
            if mode == "upsample3d" {
                self._timeConv.wrappedValue = CausalConv3d(
                    inChannels: dim, outChannels: dim * 2,
                    kernelSize: .triple(3, 1, 1), padding: .triple(1, 0, 0)
                )
            }
        } else {
            self._resample1.wrappedValue = Conv2d(
                inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2
            )
            if mode == "downsample3d" {
                self._timeConv.wrappedValue = CausalConv3d(
                    inChannels: dim, outChannels: dim,
                    kernelSize: .triple(3, 1, 1),
                    stride: .triple(2, 1, 1), padding: .triple(0, 0, 0)
                )
            }
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1)
        var t = x.dim(2)
        let h = x.dim(3), w = x.dim(4)
        var xVar = x

        if mode == "upsample3d", let tc = timeConv {
            // Temporal upsample via learned conv
            let xT = tc(xVar)  // [B, 2C, T, H, W]
            let xTR = xT.reshaped(b, 2, c, t, h, w)
            xVar = stacked([xTR[0..., 0], xTR[0..., 1]], axis: 3).reshaped(b, c, t * 2, h, w)
            t = t * 2
        }

        if mode.hasPrefix("upsample") {
            // Per-frame spatial upsample: nearest 2x + Conv2d
            xVar = xVar.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]
            xVar = repeated(xVar, count: 2, axis: 1)
            xVar = repeated(xVar, count: 2, axis: 2)
            xVar = resample1(xVar)  // Conv2d [BT, 2H, 2W, C//2]
            let cOut = xVar.dim(-1)
            return xVar.reshaped(b, t, h * 2, w * 2, cOut).transposed(0, 4, 1, 2, 3)
        } else {
            // Per-frame spatial downsample: ZeroPad(0,1,0,1) + Conv2d(stride=2)
            xVar = xVar.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]
            xVar = padded(xVar, widths: [.init(0, 0), .init(0, 1), .init(0, 1), .init(0, 0)])
            xVar = resample1(xVar)  // Conv2d stride=2
            let cOut = xVar.dim(-1)
            let hOut = xVar.dim(1), wOut = xVar.dim(2)
            xVar = xVar.reshaped(b, t, hOut, wOut, cOut).transposed(0, 4, 1, 2, 3)

            if mode == "downsample3d", let tc = timeConv {
                xVar = tc(xVar)
            }
            return xVar
        }
    }
}

// MARK: - Decoder3d

/// 3D VAE Decoder matching Wan2.1 architecture.
public class Decoder3d: Module {
    @ModuleInfo public var conv1: CausalConv3d
    @ModuleInfo public var middle: [Module]
    @ModuleInfo public var upsamples: [Module]
    public var headNorm: VaeRMSNorm
    public var headConv: CausalConv3d

    public init(
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2,
        temporalUpsample: [Bool] = [true, true, false]
    ) {
        let dims = [dim * dimMult.last!] + dimMult.reversed().map { dim * $0 }

        self._conv1.wrappedValue = CausalConv3d(
            inChannels: zDim, outChannels: dims[0], kernelSize: .int(3), padding: .int(1)
        )

        // Middle: [ResBlock, AttentionBlock, ResBlock]
        self._middle.wrappedValue = [
            ResidualBlock(inDim: dims[0], outDim: dims[0]),
            VaeAttentionBlock(dim: dims[0]),
            ResidualBlock(inDim: dims[0], outDim: dims[0]),
        ]

        // Flat upsample list
        var upsampleList = [Module]()
        for i in 0..<dims.count - 1 {
            var inD = dims[i]
            let outD = dims[i + 1]
            if [1, 2, 3].contains(i) {
                inD = inD / 2
            }
            var currentIn = inD
            for _ in 0..<(numResBlocks + 1) {
                upsampleList.append(ResidualBlock(inDim: currentIn, outDim: outD))
                currentIn = outD
            }
            if i != dimMult.count - 1 {
                let mode = temporalUpsample[i] ? "upsample3d" : "upsample2d"
                upsampleList.append(Resample(dim: outD, mode: mode))
            }
        }
        self._upsamples.wrappedValue = upsampleList

        // Output head
        self.headNorm = VaeRMSNorm(dim: dims.last!, channelFirst: true, images: false)
        self.headConv = CausalConv3d(
            inChannels: dims.last!, outChannels: 3, kernelSize: .int(3), padding: .int(1)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xVar = conv1(x)

        for layer in middle {
            if let res = layer as? ResidualBlock {
                xVar = res(xVar)
            } else if let attn = layer as? VaeAttentionBlock {
                xVar = attn(xVar)
            }
        }

        for layer in upsamples {
            if let res = layer as? ResidualBlock {
                xVar = res(xVar)
            } else if let resamp = layer as? Resample {
                xVar = resamp(xVar)
            }
        }

        xVar = silu(headNorm(xVar))
        xVar = headConv(xVar)
        return xVar
    }
}

// MARK: - Encoder3d

/// 3D VAE Encoder matching Wan2.1 architecture.
public class Encoder3d: Module {
    @ModuleInfo public var conv1: CausalConv3d
    @ModuleInfo public var downsamples: [Module]
    @ModuleInfo public var middle: [Module]
    public var headNorm: VaeRMSNorm
    public var headConv: CausalConv3d

    public init(
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2,
        temporalDownsample: [Bool] = [false, true, true]
    ) {
        let dims = [dim] + dimMult.map { dim * $0 }

        self._conv1.wrappedValue = CausalConv3d(
            inChannels: 3, outChannels: dims[0], kernelSize: .int(3), padding: .int(1)
        )

        // Flat downsample list
        var downsampleList = [Module]()
        for i in 0..<dims.count - 1 {
            var inD = dims[i]
            let outD = dims[i + 1]
            for _ in 0..<numResBlocks {
                downsampleList.append(ResidualBlock(inDim: inD, outDim: outD))
                inD = outD
            }
            if i != dimMult.count - 1 {
                let mode = temporalDownsample[i] ? "downsample3d" : "downsample2d"
                downsampleList.append(Resample(dim: outD, mode: mode))
            }
        }
        self._downsamples.wrappedValue = downsampleList

        // Middle: [ResBlock, AttentionBlock, ResBlock]
        self._middle.wrappedValue = [
            ResidualBlock(inDim: dims.last!, outDim: dims.last!),
            VaeAttentionBlock(dim: dims.last!),
            ResidualBlock(inDim: dims.last!, outDim: dims.last!),
        ]

        // Output head
        self.headNorm = VaeRMSNorm(dim: dims.last!, channelFirst: true, images: false)
        self.headConv = CausalConv3d(
            inChannels: dims.last!, outChannels: zDim, kernelSize: .int(3), padding: .int(1)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xVar = conv1(x)

        for layer in downsamples {
            if let res = layer as? ResidualBlock {
                xVar = res(xVar)
            } else if let resamp = layer as? Resample {
                xVar = resamp(xVar)
            }
        }

        for layer in middle {
            if let res = layer as? ResidualBlock {
                xVar = res(xVar)
            } else if let attn = layer as? VaeAttentionBlock {
                xVar = attn(xVar)
            }
        }

        xVar = silu(headNorm(xVar))
        xVar = headConv(xVar)
        return xVar
    }
}

// MARK: - WanVAE

/// Wan2.1 VAE wrapper with per-channel normalization.
///
/// Supports both encode (for I2V) and decode (for all models).
public class WanVAE: Module {
    public let zDim: Int
    public let mean: MLXArray
    public let std: MLXArray
    public let invStd: MLXArray

    @ModuleInfo public var conv2: CausalConv3d
    @ModuleInfo public var decoder: Decoder3d
    @ModuleInfo public var encoder: Encoder3d?
    @ModuleInfo public var conv1: CausalConv3d?

    public init(zDim: Int = 16, encoder hasEncoder: Bool = false) {
        self.zDim = zDim
        self.mean = MLXArray(vaeMean)
        self.std = MLXArray(vaeStd)
        self.invStd = 1.0 / MLXArray(vaeStd)

        self._conv2.wrappedValue = CausalConv3d(
            inChannels: zDim, outChannels: zDim, kernelSize: .int(1)
        )
        self._decoder.wrappedValue = Decoder3d(dim: 96, zDim: zDim)

        if hasEncoder {
            self._encoder.wrappedValue = Encoder3d(dim: 96, zDim: zDim * 2)
            self._conv1.wrappedValue = CausalConv3d(
                inChannels: zDim * 2, outChannels: zDim * 2, kernelSize: .int(1)
            )
        }
    }

    /// Encode video to normalized latent.
    ///
    /// - Parameter x: Video [B, 3, T, H, W] in [-1, 1]
    /// - Returns: Normalized latent [B, zDim, T_lat, H_lat, W_lat]
    public func encode(_ x: MLXArray) -> MLXArray {
        guard let enc = encoder, let c1 = conv1 else {
            fatalError("Encoder not initialized. Pass encoder: true to init.")
        }

        let out = enc(x)
        // Split mu from the conv1 output
        let convOut = c1(out)
        let mu = convOut[0..., ..<zDim]

        // Normalize: (mu - mean) * invStd
        let meanR = mean.reshaped(1, -1, 1, 1, 1)
        let invStdR = invStd.reshaped(1, -1, 1, 1, 1)
        return (mu - meanR) * invStdR
    }

    /// Decode latent to video.
    ///
    /// - Parameter z: Normalized latent [B, zDim, T, H, W]
    /// - Returns: Video [B, 3, T_out, H_out, W_out] clamped to [-1, 1]
    public func decode(_ z: MLXArray) -> MLXArray {
        let meanR = mean.reshaped(1, -1, 1, 1, 1)
        let invStdR = invStd.reshaped(1, -1, 1, 1, 1)
        let zDenorm = z / invStdR + meanR

        let x = conv2(zDenorm)
        let out = decoder(x)
        return clip(out, min: -1, max: 1)
    }
}
