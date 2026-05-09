"""SDXL generation for Art Attack."""

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    EulerAncestralDiscreteScheduler,
)

from src.config import (
    BASE_MODEL_ID,
    CONTROL_MODES,
    CONTROL_MODELS,
    DEFAULT_CONTROL_MODE,
    DEFAULT_NEGATIVE_PROMPT,
    STYLE_SUFFIX,
    VAE_MODEL_ID,
)

RESULTS_DIR = Path("results")
INPUT_PATH = RESULTS_DIR / "input_1.png"
OUTPUT_PATH = RESULTS_DIR / "output_1.png"


# Resize every input sketch to the square size expected by SDXL ControlNet.
def resize_image(image: Image.Image, size: int = 1024) -> Image.Image:
    return image.convert("RGB").resize((size, size))


# Turn the sketch into a Canny edge control image.
def make_canny_image(
    image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    image = resize_image(image, size=size)
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


# Turn dark sketch lines into a high-contrast scribble control image.
def make_scribble_image(
    image: Image.Image,
    size: int = 1024,
    threshold: int = 210,
) -> Image.Image:
    image = resize_image(image, size=size).convert("L")
    lines = np.where(np.array(image) < threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    lines = cv2.dilate(lines, kernel, iterations=1)
    return Image.fromarray(np.repeat(lines[:, :, None], 3, axis=2))


# Colors used to make stick-figure lines look closer to OpenPose output.
OPENPOSE_BODY_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]


# Convert a simple stick figure into an approximate OpenPose-style control map.
def make_pose_image(
    image: Image.Image,
    size: int = 1024,
    threshold: int = 210,
) -> Image.Image:
    image = resize_image(image, size=size).convert("L")
    lines = np.where(np.array(image) < threshold, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_line_length = max(32, size // 14)
    max_line_gap = max(12, size // 70)
    hough_threshold = max(28, size // 32)
    detected = cv2.HoughLinesP(
        lines,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    if detected is None:
        return make_scribble_image(image, size=size, threshold=threshold)

    segments = [line[0] for line in detected]
    segments.sort(
        key=lambda segment: (
            ((segment[1] + segment[3]) / 2),
            ((segment[0] + segment[2]) / 2),
        )
    )

    for index, (x1, y1, x2, y2) in enumerate(segments):
        color = OPENPOSE_BODY_COLORS[index % len(OPENPOSE_BODY_COLORS)]
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=max(4, size // 170))
        cv2.circle(canvas, (x1, y1), radius=max(3, size // 220), color=color, thickness=-1)
        cv2.circle(canvas, (x2, y2), radius=max(3, size // 220), color=color, thickness=-1)

    return Image.fromarray(canvas)


# Pick the correct control image builder for the selected mode.
def make_control_image(
    image: Image.Image,
    control_mode: str = DEFAULT_CONTROL_MODE,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    if control_mode == "none":
        raise ValueError("Control image is not used when control mode is none.")

    if control_mode == "canny":
        return make_canny_image(image, low_threshold, high_threshold, size)

    if control_mode == "scribble":
        return make_scribble_image(image, size=size)

    if control_mode == "pose":
        return make_pose_image(image, size=size)

    raise ValueError(
        f"Unknown control mode: {control_mode}. "
        f"Expected one of: {', '.join(CONTROL_MODES)}"
    )


# Build the final model prompt from the optional LoRA trigger, user prompt, and fixed style.
def build_prompt(prompt: str, lora_trigger: str = "") -> str:
    parts = [part.strip() for part in (lora_trigger or "", prompt or "", STYLE_SUFFIX)]
    return ", ".join(part for part in parts if part)


# Load SDXL, optionally with the selected ControlNet, and an optional LoRA.
def load_pipeline(
    lora_path: str | None = None,
    control_mode: str = DEFAULT_CONTROL_MODE,
):
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"Unknown control mode: {control_mode}. "
            f"Expected one of: {', '.join(CONTROL_MODES)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None

    print(f"Using device: {device}")

    print(f"Loading VAE: {VAE_MODEL_ID}")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=dtype,
    )

    pipeline_class = StableDiffusionXLPipeline
    pipeline_kwargs = {}

    if control_mode != "none":
        controlnet_model_id = CONTROL_MODELS[control_mode]
        print(f"Loading {control_mode} ControlNet: {controlnet_model_id}")
        pipeline_class = StableDiffusionXLControlNetPipeline
        pipeline_kwargs["controlnet"] = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
        )

    print(f"Loading SDXL base model: {BASE_MODEL_ID}")
    pipe = pipeline_class.from_pretrained(
        BASE_MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
        variant=variant,
        use_safetensors=True,
        **pipeline_kwargs,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception as exc:
            print(f"xFormers not enabled: {exc}")

    if lora_path:
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="artattack")
        pipe._artattack_lora_loaded = True

    return pipe


# Save the latest input and output only, overwriting the previous test result.
def save_latest_result(sketch: Image.Image | None, output: Image.Image) -> Tuple[Path | None, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = None

    if sketch is not None:
        sketch.convert("RGB").save(INPUT_PATH)
        input_path = INPUT_PATH
    elif INPUT_PATH.exists():
        INPUT_PATH.unlink()

    output.save(OUTPUT_PATH)
    return input_path, OUTPUT_PATH


# Generate one image and return the generated image, optional control image, and final prompt.
def generate_image(
    pipe,
    sketch: Image.Image | None,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 35,
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 0.9,
    seed: int = 42,
    size: int = 1024,
    low_threshold: int = 80,
    high_threshold: int = 180,
    lora_trigger: str = "",
    lora_scale: float = 1.0,
    control_mode: str = DEFAULT_CONTROL_MODE,
) -> Tuple[Image.Image, Image.Image | None, str]:
    final_prompt = build_prompt(prompt, lora_trigger=lora_trigger)
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    if getattr(pipe, "_artattack_lora_loaded", False) and hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["artattack"], adapter_weights=[float(lora_scale)])

    generation_kwargs = {
        "prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
        "width": size,
        "height": size,
    }

    control_image = None
    if control_mode != "none":
        control_image = make_control_image(
            sketch,
            control_mode=control_mode,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            size=size,
        )
        generation_kwargs["image"] = control_image
        generation_kwargs["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)

    output = pipe(**generation_kwargs).images[0]

    return output, control_image, final_prompt


# Save generated output files when the script is run from the command line.
def save_generation(
    pipe,
    sketch_path: str | None,
    prompt: str,
    **kwargs,
):
    control_mode = kwargs.get("control_mode", DEFAULT_CONTROL_MODE)
    sketch = None if control_mode == "none" else Image.open(sketch_path)

    image, control_image, final_prompt = generate_image(
        pipe=pipe,
        sketch=sketch,
        prompt=prompt,
        **kwargs,
    )

    input_path, output_path = save_latest_result(sketch, image)

    print("Generation complete.")
    print(f"Final prompt: {final_prompt}")
    if input_path:
        print(f"Input image saved to: {input_path}")
    print(f"Generated image saved to: {output_path}")


# Define the command-line options for non-Gradio generation.
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Art Attack SDXL dark fantasy character art.")
    parser.add_argument("--sketch", default="")
    parser.add_argument("--prompt", default="full body dark fantasy knight holding a glowing sword")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--control-scale", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--low-threshold", type=int, default=80)
    parser.add_argument("--high-threshold", type=int, default=180)
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-trigger", default="")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    return parser.parse_args()


# Run one command-line generation if this file is executed directly.
if __name__ == "__main__":
    args = parse_args()

    if args.control_mode != "none" and not os.path.exists(args.sketch):
        raise FileNotFoundError(f"Sketch not found: {args.sketch}")

    pipe = load_pipeline(
        lora_path=args.lora_path or None,
        control_mode=args.control_mode,
    )

    save_generation(
        pipe=pipe,
        sketch_path=args.sketch,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        seed=args.seed,
        size=args.size,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        control_mode=args.control_mode,
        lora_trigger=args.lora_trigger,
        lora_scale=args.lora_scale,
    )
