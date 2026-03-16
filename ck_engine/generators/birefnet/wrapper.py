"""BiRefNet wrapper for generating coarse alpha hints.

Uses ZhengPeng7/BiRefNet salient object segmentation model to produce
per-frame foreground masks. These masks serve as coarse alpha hints
for the CorridorKey inference pipeline.

VRAM: ~4.8 GB FP32, ~3.5 GB FP16 at 1024x1024.
License: MIT.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from ck_engine.config import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD, VIDEO_EXTS

logger = logging.getLogger(__name__)

MODEL_ID = "ZhengPeng7/BiRefNet"
IMAGE_SIZE = (1024, 1024)


class BiRefNetProcessor:
    """Thin wrapper around BiRefNet for batch alpha hint generation."""

    def __init__(self, device: str = "cpu") -> None:
        from transformers import AutoModelForImageSegmentation

        self.device = torch.device(device)

        logger.info("Loading BiRefNet model from %s ...", MODEL_ID)
        logger.warning(
            "BiRefNet requires trust_remote_code=True: custom model code will be "
            "downloaded and executed from %s. Verify the repository before running "
            "in sensitive environments.",
            MODEL_ID,
        )
        torch.set_float32_matmul_precision("high")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.model.eval().to(self.device)

        # Detect the dtype the model weights were stored in (BiRefNet
        # ships as float16) so we can cast inputs to match.
        first_param = next(self.model.parameters())
        self.dtype = first_param.dtype

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(IMAGE_SIZE, antialias=True),
                transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
            ]
        )
        logger.info("BiRefNet loaded on %s (dtype=%s).", self.device, self.dtype)

    @torch.no_grad()
    def process_frame(self, img_rgb: np.ndarray) -> np.ndarray:
        """Process a single RGB uint8 frame and return a grayscale alpha mask.

        Parameters
        ----------
        img_rgb : np.ndarray
            Input image in RGB uint8 format, shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Grayscale mask in uint8, shape (H, W), values 0-255.
        """
        from PIL import Image

        orig_h, orig_w = img_rgb.shape[:2]

        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        preds = self.model(tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Resize back to original dimensions
        mask = pred.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        mask = mask.squeeze().clamp(0, 1)

        return (mask.numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# AlphaGenerator wrapper
# ---------------------------------------------------------------------------

class BiRefNetAlphaGenerator:
    """AlphaGenerator wrapper around BiRefNetProcessor."""

    name = "birefnet"
    is_temporal = False
    requires_mask = False

    def __init__(self, device: str = "cpu") -> None:
        self._processor = BiRefNetProcessor(device=device)

    # -- private helpers ----------------------------------------------------

    def _iter_frames(
        self,
        input_path: Path,
        indices: set[int] | None = None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(stem, rgb)`` tuples lazily, one frame at a time."""
        is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS

        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            try:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if indices is not None and idx not in indices:
                        idx += 1
                        continue
                    yield (f"{idx:04d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    idx += 1
            finally:
                cap.release()
        else:
            files = sorted(
                f for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
            for i, f in enumerate(files):
                if indices is not None and i not in indices:
                    continue

                if f.suffix.lower() == ".exr":
                    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img = np.clip(img, 0.0, 1.0)
                        img = img ** (1.0 / 2.2)
                        img = (img * 255.0).astype(np.uint8)
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.imread(str(f))

                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    yield (f.stem, rgb)

    def _count_frames(
        self,
        input_path: Path,
        indices: set[int] | None = None,
    ) -> int:
        """Return frame count without loading pixel data."""
        is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS

        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if indices is not None:
                return len(indices.intersection(range(total)))
            return total
        else:
            total = sum(
                1 for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
            if indices is not None:
                return len(indices.intersection(range(total)))
            return total

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        input_dir: str,
        output_dir: str,
        *,
        mask_dir: str | None = None,
        frame_indices: Sequence[int] | None = None,
        skip_existing: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        os.makedirs(output_dir, exist_ok=True)
        input_path = Path(input_dir)
        indices_set = set(frame_indices) if frame_indices is not None else None
        total = self._count_frames(input_path, indices_set)

        if total == 0:
            logger.warning("No frames found at %s", input_path)
            return 0

        written = 0
        for i, (stem, rgb) in enumerate(self._iter_frames(input_path, indices_set)):
            out_path = os.path.join(output_dir, f"{stem}.png")
            if skip_existing and os.path.exists(out_path):
                if on_progress:
                    on_progress(i + 1, total)
                continue
            mask = self._processor.process_frame(rgb)
            cv2.imwrite(out_path, mask)
            written += 1
            if on_progress:
                on_progress(i + 1, total)

        return written
