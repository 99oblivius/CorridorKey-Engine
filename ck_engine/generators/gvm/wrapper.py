from __future__ import annotations

import os
import os.path as osp
import cv2
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

# Relative imports from the internal gvm package
# Assuming this file is inside alpha_generators/gvm/
from .gvm.pipelines.pipeline_gvm import GVMPipeline
from .gvm.utils.inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from .gvm.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from ck_engine.config import IMAGE_EXTS, VIDEO_EXTS

# GVM-specific resolution constants (model-internal, not project-wide)
_GVM_MIN_HEIGHT = 1024
_GVM_MAX_WIDTH = 1920
_GVM_FALLBACK_HEIGHT = 1080
_GVM_FALLBACK_WIDTH = 1920
_GVM_ALPHA_UPPER = 240.0 / 255.0
_GVM_ALPHA_LOWER = 25.0 / 255.0


def impad_multi(img, multiple=32):
    # img: (N, C, H, W)
    h, w = img.shape[2], img.shape[3]
    
    target_h = int(np.ceil(h / multiple) * multiple)
    target_w = int(np.ceil(w / multiple) * multiple)

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # F.pad expects (padding_left, padding_right, padding_top, padding_bottom)
    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return padded, (pad_top, pad_left, pad_bottom, pad_right)

def sequence_collate_fn(examples):
    rgb_values = torch.stack([example["image"] for example in examples])
    rgb_values = rgb_values.to(memory_format=torch.contiguous_format).float()
    rgb_names = [example["filename"] for example in examples]
    return {'rgb_values': rgb_values, 'rgb_names': rgb_names}

class GVMProcessor:
    def __init__(self, 
                 model_base=None,
                 unet_base=None,
                 lora_base=None,
                 device="cpu",
                 seed=42):
        self.device = torch.device(device)

        # Resolve default weights path relative to this file
        if model_base is None:
            model_base = osp.join(osp.dirname(__file__), "weights")

        self.model_base = model_base
        self.unet_base = unet_base
        self.lora_base = lora_base

        # Scoped RNG — no global state mutation
        gen_device = self.device if str(self.device).startswith("cuda") else "cpu"
        self._generator = torch.Generator(device=gen_device)
        self._generator.manual_seed(seed)
        
        logging.info(f"Loading GVM models from {model_base}...")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(model_base, subfolder="vae", torch_dtype=torch.float16)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_base, subfolder="scheduler")
        
        unet_folder = unet_base if unet_base is not None else model_base
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_folder, 
            subfolder="unet", 
            class_embed_type=None,
            torch_dtype=torch.float16
        )

        self.pipe = GVMPipeline(vae=self.vae, unet=self.unet, scheduler=self.scheduler)
        if lora_base:
            self.pipe.load_lora_weights(lora_base)
                
        self.pipe = self.pipe.to(self.device, dtype=torch.float16)
        self.pipe._generator = self._generator
        logging.info("Models loaded.")

    def process_sequence(self, input_path, output_dir, 
                         num_frames_per_batch=8,
                         denoise_steps=1,
                         max_frames=None,
                         decode_chunk_size=8,
                         num_interp_frames=1,
                         num_overlap_frames=1,
                         use_clip_img_emb=False,
                         noise_type='zeros',
                         mode='matte',
                         write_video=True,
                         direct_output_dir=None):
        """
        Process a single video or directory of images.
        """
        input_path = Path(input_path)
        file_name = input_path.stem
        is_video = input_path.suffix.lower() in VIDEO_EXTS
        
        # --- Determine Resolution & Upscaling ---
        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            try:
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
        else:
            image_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
            if not image_files:
                logging.warning(f"No images found in {input_path}")
                return
            # Use cv2 for EXR support if needed
            first_img_path = str(image_files[0])
            if first_img_path.lower().endswith('.exr'):
                 # import cv2 # Global import used
                 if "OPENCV_IO_ENABLE_OPENEXR" not in os.environ:
                     os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                 img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
            else:
                 img = cv2.imread(first_img_path)
                 
            if img is not None:
                orig_h, orig_w = img.shape[:2]
            else:
                orig_h, orig_w = _GVM_FALLBACK_HEIGHT, _GVM_FALLBACK_WIDTH

        target_h = orig_h
        if target_h < _GVM_MIN_HEIGHT:
            target_h = _GVM_MIN_HEIGHT

        # Calculate max resolution / long edge
        if orig_h < orig_w: # Landscape
            ratio = orig_w / orig_h
            new_long = int(_GVM_MIN_HEIGHT * ratio)
        else:
            ratio = orig_h / orig_w
            new_long = int(_GVM_MIN_HEIGHT * ratio)

        if new_long > _GVM_MAX_WIDTH:
            new_long = _GVM_MAX_WIDTH

        max_res_param = new_long

        transform = Compose([
            ToTensor(),
            Resize(size=_GVM_MIN_HEIGHT, max_size=max_res_param, antialias=True)
        ])

        if is_video:
            reader = VideoReader(
                str(input_path), 
                max_frames=max_frames,
                transform=transform
            )
        else:
            reader = ImageSequenceReader(
                str(input_path), 
                transform=transform
            )

        # Get upscaled shape from first frame
        first_frame = reader[0]
        if isinstance(first_frame, dict):
             first_frame = first_frame['image']
        
        current_upscaled_shape = list(first_frame.shape[1:]) # H, W
        if current_upscaled_shape[0] % 2 != 0: current_upscaled_shape[0] -= 1
        if current_upscaled_shape[1] % 2 != 0: current_upscaled_shape[1] -= 1
        current_upscaled_shape = tuple(current_upscaled_shape)

        # Output preparation
        fps = reader.frame_rate if hasattr(reader, 'frame_rate') else 24.0
        
        if direct_output_dir:
            # Write directly to this folder
            os.makedirs(direct_output_dir, exist_ok=True)
            writer_alpha_seq = ImageSequenceWriter(direct_output_dir, extension='png')
            writer_alpha = None
            if write_video:
                 # Warning: direct mode might not support video naming nicely without logic
                 # Let's write video into the directory with fixed name
                 writer_alpha = VideoWriter(osp.join(direct_output_dir, f"{file_name}_alpha.mp4"), frame_rate=fps)
        else:
            # Create output directory for this specific file
            file_output_dir = osp.join(output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)
            logging.info(f"Processing {input_path} -> {file_output_dir}")
            
            writer_alpha = VideoWriter(osp.join(file_output_dir, f"{file_name}_alpha.mp4"), frame_rate=fps) if write_video else None
            writer_alpha_seq = ImageSequenceWriter(osp.join(file_output_dir, "alpha_seq"), extension='png')
        
        # Dataloader
        if is_video:
            dataloader = DataLoader(reader, batch_size=num_frames_per_batch)
        else:
            dataloader = DataLoader(reader, batch_size=num_frames_per_batch, collate_fn=sequence_collate_fn)

        upper_bound = _GVM_ALPHA_UPPER
        lower_bound = _GVM_ALPHA_LOWER

        try:
            for batch_id, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Inferencing {file_name}"):
                filenames = []
                if is_video:
                    b, _, h, w = batch.shape
                    for i in range(b):
                        file_id = batch_id * b + i
                        filenames.append(f"{file_id:05d}.jpg")
                else:
                    filenames = batch['rgb_names']
                    batch = batch['rgb_values']

                # Pad (Reflective)
                batch, pad_info = impad_multi(batch)

                # Inference
                with torch.no_grad():
                    pipe_out = self.pipe(
                        batch.to(self.device, dtype=torch.float16),
                        num_frames=num_frames_per_batch,
                        num_overlap_frames=num_overlap_frames,
                        num_interp_frames=num_interp_frames,
                        decode_chunk_size=decode_chunk_size,
                        num_inference_steps=denoise_steps,
                        mode=mode,
                        use_clip_img_emb=use_clip_img_emb,
                        noise_type=noise_type,
                        ensemble_size=1,
                    )
                image = pipe_out.image
                alpha = pipe_out.alpha

                # Crop padding
                out_h, out_w = image.shape[2:]
                pad_t, pad_l, pad_b, pad_r = pad_info

                end_h = out_h - pad_b
                end_w = out_w - pad_r

                image = image[:, :, pad_t:end_h, pad_l:end_w]
                alpha = alpha[:, :, pad_t:end_h, pad_l:end_w]

                # Resize to ensure exact match if there's any discrepancy
                alpha = F.interpolate(alpha, current_upscaled_shape, mode='bilinear')

                # Threshold
                alpha[alpha>=upper_bound] = 1.0
                alpha[alpha<=lower_bound] = 0.0

                if writer_alpha: writer_alpha.write(alpha)
                writer_alpha_seq.write(alpha, filenames=filenames)
        finally:
            if writer_alpha: writer_alpha.close()
            writer_alpha_seq.close()
        logging.info(f"Finished {file_name}")


# ---------------------------------------------------------------------------
# AlphaGenerator wrapper
# ---------------------------------------------------------------------------

class GVMAlphaGenerator:
    """AlphaGenerator wrapper around GVMProcessor."""

    name = "gvm"
    is_temporal = True
    requires_mask = False

    def __init__(self, device: str = "cpu") -> None:
        self._processor = GVMProcessor(device=device)

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

        # When skip_existing is requested, check whether all expected output
        # files already exist.  GVM is a temporal model — partial re-runs
        # would break temporal consistency — so the granularity is all-or-nothing.
        if skip_existing:
            input_path_check = Path(input_dir)
            is_video_check = input_path_check.is_file() and input_path_check.suffix.lower() in VIDEO_EXTS
            if is_video_check:
                cap = cv2.VideoCapture(str(input_path_check))
                try:
                    expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
            else:
                expected_count = sum(
                    1 for f in os.listdir(input_dir)
                    if Path(f).suffix.lower() in IMAGE_EXTS
                )
            if frame_indices is not None:
                expected_count = len(frame_indices)

            existing = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".png"))
            if len(existing) >= expected_count > 0:
                logging.info(
                    "skip_existing: all %d output files already present in %s, skipping GVM.",
                    expected_count,
                    output_dir,
                )
                total = sum(1 for f in os.listdir(output_dir) if f.lower().endswith(".png"))
                if on_progress:
                    on_progress(total, total)
                return total

        self._processor.process_sequence(
            input_path=input_dir,
            output_dir=None,
            num_frames_per_batch=1,
            decode_chunk_size=1,
            denoise_steps=1,
            mode="matte",
            write_video=False,
            direct_output_dir=output_dir,
        )

        # Rename GVM output files to {input_stem}.png
        generated = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".png"))
        input_path = Path(input_dir)
        is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS

        if is_video:
            stems = [f"{i:05d}" for i in range(len(generated))]
        elif input_path.is_dir():
            stems = [
                Path(f).stem
                for f in sorted(os.listdir(input_dir))
                if Path(f).suffix.lower() in IMAGE_EXTS
            ]
        else:
            stems = [f"{i:05d}" for i in range(len(generated))]

        for i, gvm_file in enumerate(generated):
            if i >= len(stems):
                break
            new_name = f"{stems[i]}.png"
            old_path = os.path.join(output_dir, gvm_file)
            new_path = os.path.join(output_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)

        # Filter by frame_indices if specified (temporal: delete non-selected)
        if frame_indices is not None:
            indices_set = set(frame_indices)
            all_pngs = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".png"))
            for i, fname in enumerate(all_pngs):
                if i not in indices_set:
                    os.remove(os.path.join(output_dir, fname))

        total = sum(1 for f in os.listdir(output_dir) if f.lower().endswith(".png"))
        if on_progress:
            on_progress(total, total)
        return total
