from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..optimization_config import OptimizationConfig

# ---------------------------------------------------------------------------
# Shared utility: patch input layer for non-3-channel inputs
# ---------------------------------------------------------------------------


def patch_input_layer(encoder: nn.Module, in_channels: int) -> None:
    """Modify the first convolution layer to accept *in_channels*.

    Copies existing RGB weights and zero-initialises the extra channels.
    Works with timm Hiera models (tries ``encoder.model.patch_embed.proj``
    first, then ``encoder.patch_embed.proj``).
    """
    try:
        proj = encoder.model.patch_embed.proj
    except AttributeError:
        proj = encoder.patch_embed.proj

    weight = proj.weight.data  # [Out, 3, K, K]
    bias = proj.bias.data if proj.bias is not None else None
    out_channels, _, k, _ = weight.shape

    new_conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=k,
        stride=proj.stride,
        padding=proj.padding,
        bias=(bias is not None),
    )

    new_conv.weight.data[:, :3, :, :] = weight
    new_conv.weight.data[:, 3:, :, :] = 0.0
    if bias is not None:
        new_conv.bias.data = bias

    try:
        encoder.model.patch_embed.proj = new_conv
    except AttributeError:
        encoder.patch_embed.proj = new_conv

    print(f"Patched input layer: 3 channels -> {in_channels} channels (Extra initialized to 0)")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Linear embedding: C_in -> C_out."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    def __init__(
        self, feature_channels: list[int] | None = None, embedding_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        # MLP layers to unify channel dimensions
        self.linear_c4 = MLP(input_dim=feature_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=feature_channels[0], embed_dim=embedding_dim)

        # Fuse
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        # Predict
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features

        n, _, h, w = c4.shape

        # Resize to C1 size (which is H/4)
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.bn(_c)
        _c = self.relu(_c)

        x = self.dropout(_c)
        x = self.classifier(x)

        return x


class RefinerBlock(nn.Module):
    """Residual Block with Dilation and GroupNorm (Safe for Batch Size 2)."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNRefinerModule(nn.Module):
    """Dilated Residual Refiner (Receptive Field ~65px).

    Designed to solve Macroblocking artifacts from Hiera.
    Structure: Stem -> Res(d1) -> Res(d2) -> Res(d4) -> Res(d8) -> Projection.
    """

    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated Residual Blocks (RF Expansion)
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)

        # Final Projection (No Activation, purely additive logits)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Tiny Noise Init (Whisper) - Provides gradients without shock
        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.final.bias, 0)

    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W]
        # coarse_pred: [B, 4, H, W]
        x = torch.cat([img, coarse_pred], dim=1)

        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Output Scaling (10x Boost)
        # Allows the Refiner to predict small stable values (e.g. 0.5) that become strong corrections (5.0).
        return self.final(x) * 10.0


# ---------------------------------------------------------------------------
# GreenFormer
# ---------------------------------------------------------------------------


class GreenFormer(nn.Module):
    """Hiera-based green screen keying model.

    Accepts an optional :class:`OptimizationConfig` to selectively apply
    VRAM optimizations (FlashAttention patching, tiled refiner, cache
    clearing).  When *optimization_config* is ``None``, behaviour is
    identical to the original unoptimized model.
    """

    def __init__(
        self,
        encoder_name: str = "hiera_base_plus_224.mae_in1k_ft_in1k",
        in_channels: int = 4,
        img_size: int = 512,
        use_refiner: bool = True,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = optimization_config or OptimizationConfig()
        self.img_size = img_size
        self.in_channels = in_channels

        # --- Encoder ---
        print(f"Initializing {encoder_name} (img_size={img_size})...")
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, img_size=img_size)
        print("Skipped downloading base weights (relying on custom checkpoint).")

        # Patch first layer for 4 channels
        if in_channels != 3:
            patch_input_layer(self.encoder, in_channels)

        # FlashAttention patching (conditional)
        if self.config.flash_attention:
            from .optimized_model import _patch_hiera_global_attention

            n_patched = _patch_hiera_global_attention(self.encoder.model)
            if n_patched:
                print(f"[Optimized] Patched {n_patched} global-attention blocks for FlashAttention.")

        # Get feature info
        try:
            feature_channels = self.encoder.feature_info.channels()
        except (AttributeError, TypeError):
            feature_channels = [112, 224, 448, 896]
        print(f"Feature Channels: {feature_channels}")
        self._feature_channels = feature_channels

        # --- Hiera internals (needed by subclass for token routing) ---
        hiera = self.encoder.model
        self._stage_ends = list(hiera.stage_ends)
        self._patch_stride = hiera.patch_stride

        tokens_h = img_size // self._patch_stride[0]
        tokens_w = img_size // self._patch_stride[1]
        self._stage_token_shapes: list[tuple[int, int]] = []
        for i in range(len(self._stage_ends)):
            self._stage_token_shapes.append((tokens_h, tokens_w))
            if i < hiera.q_pool:
                tokens_h //= hiera.q_stride[0]
                tokens_w //= hiera.q_stride[1]

        # --- Decoders ---
        embedding_dim = 256
        self.alpha_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=1)
        self.fg_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=3)

        # --- Refiner ---
        self.use_refiner = use_refiner
        if self.use_refiner:
            if self.config.tiled_refiner:
                from .optimized_model import TiledCNNRefiner

                self.refiner = TiledCNNRefiner(
                    in_channels=7,
                    hidden_channels=64,
                    out_channels=4,
                    tile_size=self.config.tile_size,
                    tile_overlap=self.config.tile_overlap,
                )
                print(
                    f"[Optimized] Using TiledCNNRefiner "
                    f"(tile={self.config.tile_size}, overlap={self.config.tile_overlap})."
                )
            else:
                self.refiner = CNNRefinerModule(in_channels=7, hidden_channels=64, out_channels=4)
        else:
            self.refiner = None
            print("Refiner Module DISABLED (Backbone Only Mode).")

    # kept for backward compat; delegates to the module-level function
    def _patch_input_layer(self, in_channels: int) -> None:
        patch_input_layer(self.encoder, in_channels)

    # ------------------------------------------------------------------
    # Decode + Refine pipeline (shared with subclasses)
    # ------------------------------------------------------------------

    def _decode_and_refine(
        self,
        features: list[torch.Tensor],
        x: torch.Tensor,
        input_size: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        """Shared decode -> upsample -> sigmoid -> refine -> sigmoid.

        Called by :meth:`forward` and overridden encoder paths in
        ``OptimizedGreenFormer``.
        """
        # Decode
        alpha_logits = self.alpha_decoder(features)
        fg_logits = self.fg_decoder(features)

        # Upsample to full resolution
        alpha_logits_up = F.interpolate(alpha_logits, size=input_size, mode="bilinear", align_corners=False)
        fg_logits_up = F.interpolate(fg_logits, size=input_size, mode="bilinear", align_corners=False)

        # Coarse probs (for refiner input)
        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)

        # Cache clearing before refiner
        if self.config.cache_clearing and x.is_cuda:
            torch.cuda.empty_cache()

        # Refine
        rgb = x[:, :3, :, :]
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)

        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb, coarse_pred)
        else:
            delta_logits = torch.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, 0:1]
        delta_fg = delta_logits[:, 1:4]

        # Residual addition in logit space
        alpha_final_logits = alpha_logits_up + delta_alpha
        fg_final_logits = fg_logits_up + delta_fg

        # Final activation
        alpha_final = torch.sigmoid(alpha_final_logits)
        fg_final = torch.sigmoid(fg_final_logits)

        return {"alpha": alpha_final, "fg": fg_final}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        # Encode
        features = self.encoder(x)

        # Cache clearing between encoder and decoder
        if self.config.cache_clearing and x.is_cuda:
            torch.cuda.empty_cache()

        return self._decode_and_refine(features, x, input_size)
