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
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

MODEL_ID = "ZhengPeng7/BiRefNet"
IMAGE_SIZE = (1024, 1024)


class BiRefNetProcessor:
    """Thin wrapper around BiRefNet for batch alpha hint generation."""

    def __init__(self, device: str = "cpu") -> None:
        from transformers import AutoModelForImageSegmentation

        self.device = torch.device(device)

        logger.info("Loading BiRefNet model from %s ...", MODEL_ID)
        torch.set_float32_matmul_precision("high")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    def process_sequence(
        self,
        input_path: str,
        output_dir: str,
    ) -> int:
        """Process an image sequence or video, writing masks to output_dir.

        Parameters
        ----------
        input_path : str
            Path to a directory of images or a video file.
        output_dir : str
            Directory to write output PNG masks into.

        Returns
        -------
        int
            Number of frames processed.
        """
        from tqdm import tqdm

        input_path = Path(input_path)
        os.makedirs(output_dir, exist_ok=True)

        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"}
        VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

        is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS

        frames: list[tuple[str, np.ndarray]] = []

        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((f"{idx:04d}", rgb))
                idx += 1
            cap.release()
        else:
            files = sorted(f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            for f in files:
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
                    frames.append((f.stem, rgb))

        if not frames:
            logger.warning("No frames found at %s", input_path)
            return 0

        total = 0
        for stem, rgb in tqdm(frames, desc="BiRefNet"):
            mask = self.process_frame(rgb)
            out_path = os.path.join(output_dir, f"{stem}.png")
            cv2.imwrite(out_path, mask)
            total += 1

        return total
