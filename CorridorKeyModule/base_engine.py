"""Shared base class for CorridorKey inference engines.

All engine variants (original, optimized, MLX adapter) share the same
``process_frame()`` pipeline and checkpoint-loading logic.  Subclasses
only need to implement :meth:`_create_model` to return the appropriate
model variant.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import color_utils as cu
from .optimization_config import OptimizationConfig, PerformanceMetrics


class _BaseCorridorKeyEngine(ABC):
    """Shared inference engine logic for all Torch-based backends.

    Subclasses must implement :meth:`_create_model` to return the
    appropriate ``GreenFormer`` variant.  Everything else -- constructor
    setup, checkpoint loading, ``process_frame()`` -- lives here.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.config = optimization_config or OptimizationConfig()

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Engine-level optimization: disable cuDNN benchmark
        if self.config.disable_cudnn_benchmark and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            print("[Optimized] cuDNN benchmark disabled (saves 2-5 GB workspace).")

        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Instantiate and return the model (not yet loaded with weights).

        The returned model will be moved to ``self.device``, set to
        ``eval()`` mode, and loaded with the checkpoint by
        :meth:`_load_model`.
        """
        ...

    def _report_load_results(self, missing: list[str], unexpected: list[str]) -> None:
        """Log missing / unexpected state-dict keys after loading.

        The default implementation warns about any missing or unexpected
        keys.  ``OptimizedCorridorKeyEngine`` overrides this to handle
        expected LTRM keys gracefully.
        """
        if missing:
            print(f"[Warning] Missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")

    # ------------------------------------------------------------------
    # Checkpoint loading (shared)
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        """Load the checkpoint into the model returned by :meth:`_create_model`."""
        print(f"Loading CorridorKey from {self.checkpoint_path}...")

        model = self._create_model()
        model = model.to(self.device)
        model.eval()

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix compiled-model prefix & handle PosEmbed mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            # PosEmbed interpolation
            if "pos_embed" in k and k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Resizing {k} from {v.shape} to {model_state[k].shape}")
                    N_src = v.shape[1]
                    C = v.shape[2]
                    grid_src = int(math.sqrt(N_src))
                    N_dst = model_state[k].shape[1]
                    grid_dst = int(math.sqrt(N_dst))

                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        self._report_load_results(missing, unexpected)

        # Report VRAM after loading
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"Model loaded. GPU memory: {allocated:.2f} GB")

        return model

    # ------------------------------------------------------------------
    # Frame processing (shared -- THE single implementation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Process a single frame.

        Args:
            image: ``[H, W, 3]`` numpy array (0-1 float or 0-255 uint8).
                sRGB by default; linear if *input_is_linear* is True.
            mask_linear: ``[H, W]`` or ``[H, W, 1]`` numpy array (0-1).
                The coarse alpha-hint mask.
            refiner_scale: Multiplier for refiner output deltas.
            input_is_linear: If True, resizes in linear then converts to sRGB.
            fg_is_straight: If True, foreground is straight (unpremultiplied).
            despill_strength: Green-spill removal strength (0-1).
            auto_despeckle: Remove small disconnected alpha components.
            despeckle_size: Minimum pixel area to keep.

        Returns:
            Dictionary with keys ``"alpha"``, ``"fg"``, ``"comp"``,
            ``"processed"``, and optionally ``"metrics"``.
        """
        metrics = PerformanceMetrics() if self.config.enable_metrics else None

        # === 1. Input normalization ===
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # === 2. Resize to model size ===
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # === 3. ImageNet normalization ===
        img_norm = (img_resized - self.mean) / self.std

        # === 4. Prepare tensor ===
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)  # [H, W, 4]
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        # === 5. Inference ===
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        if metrics is not None:
            ctx = metrics.measure("inference", self.device)
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.model(inp_t)

        if handle:
            handle.remove()

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]

        # === 6. Post-process (resize back to original resolution) ===
        if metrics is not None:
            ctx_post = metrics.measure("postprocess", self.device)
        else:
            ctx_post = nullcontext()

        with ctx_post:
            res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
            res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
            res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
            res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

            if res_alpha.ndim == 2:
                res_alpha = res_alpha[:, :, np.newaxis]

            # A. Clean matte (auto-despeckle)
            if auto_despeckle:
                processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
            else:
                processed_alpha = res_alpha

            # B. Despill FG
            fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)

            # C. Premultiply for EXR (convert to linear first)
            fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
            fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

            # D. Pack RGBA (all linear float)
            processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

            # E. Composite on checkerboard
            bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
            bg_lin = cu.srgb_to_linear(bg_srgb)

            if fg_is_straight:
                comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
            else:
                comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

            comp_srgb = cu.linear_to_srgb(comp_lin)

        # === Peak VRAM reporting ===
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            print(f"Peak VRAM: {peak:.2f} GB")

        # === Finalize metrics ===
        if metrics is not None:
            metrics.finalize(self.device)

        result: dict[str, np.ndarray | PerformanceMetrics] = {
            "alpha": res_alpha,
            "fg": res_fg,
            "comp": comp_srgb,
            "processed": processed_rgba,
        }

        if metrics is not None:
            result["metrics"] = metrics

        return result
