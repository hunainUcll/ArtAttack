"""
Sketch2DarkFantasy - SDXL + sketch-conditioned generation module.

Pipeline:
Sketch image -> line preprocessing -> SDXL ControlNet -> dark fantasy output
"""

import argparse
import os
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROL_MODELS = {
    "scribble": "xinsir/controlnet-scribble-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
}
DEFAULT_CONTROL_MODE: Literal["scribble", "canny"] = "scribble"

# Optional better VAE for SDXL stability
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

STYLE_SUFFIX = (
    "dark fantasy character concept art, full body character, highly detailed, "
    "realistic, cinematic lighting, dramatic atmosphere, sharp focus, "
    "professional fantasy illustration, detailed armor, dark medieval background"
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, pixelated, bad anatomy, bad hands, "
    "extra fingers, missing fingers, extra limbs, missing limbs, deformed body, "
    "distorted face, ugly face, cropped, out of frame, text, watermark, logo, "
    "cartoon, childish, stick figure, doodle, simple line art"
)


# ------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    return torch.float16 if get_device() == "cuda" else torch.float32


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

def resize_image(image: Image.Image, size: int = 1024) -> Image.Image:
    """
    Resize image to square SDXL resolution.
    SDXL works best at 1024x1024.
    """
    return image.convert("RGB").resize((size, size))


def make_canny_image(
    image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    """
    Converts sketch to Canny edge image for ControlNet.
    """
    image = resize_image(image, size=size)

    image_np = np.array(image)
    edges = cv2.Canny(image_np, low_threshold, high_threshold)

    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)

    return Image.fromarray(edges)


def make_scribble_image(
    image: Image.Image,
    size: int = 1024,
    threshold: int = 210,
) -> Image.Image:
    """
    Converts rough black-on-white stick sketches into white lines on black.

    This usually preserves the intended pose better than Canny for stick figures,
    because Canny tends to detect both sides of each drawn stroke.
    """
    image = resize_image(image, size=size).convert("L")
    image_np = np.array(image)

    lines = np.where(image_np < threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    lines = cv2.dilate(lines, kernel, iterations=1)

    lines = lines[:, :, None]
    lines = np.concatenate([lines, lines, lines], axis=2)

    return Image.fromarray(lines)


def make_control_image(
    image: Image.Image,
    control_mode: str = DEFAULT_CONTROL_MODE,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    if control_mode == "canny":
        return make_canny_image(
            image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            size=size,
        )

    if control_mode == "scribble":
        return make_scribble_image(image, size=size)

    raise ValueError(
        f"Unknown control mode: {control_mode}. "
        f"Expected one of: {', '.join(CONTROL_MODELS)}"
    )


# ------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------

def build_prompt(prompt: str, lora_trigger: str = "") -> str:
    if prompt is None:
        prompt = ""

    if lora_trigger is None:
        lora_trigger = ""

    prompt = prompt.strip()
    lora_trigger = lora_trigger.strip()

    parts = []

    if lora_trigger:
        parts.append(lora_trigger)

    if prompt:
        parts.append(prompt)

    parts.append(STYLE_SUFFIX)

    return ", ".join(parts)


# ------------------------------------------------------------
# Pipeline loading
# ------------------------------------------------------------

def load_pipeline(
    lora_path: str | None = None,
    control_mode: str = DEFAULT_CONTROL_MODE,
):
    """
    Loads SDXL + SDXL ControlNet.

    This is heavier than SD 1.5, but your RTX 4070 Ti Super should handle it.
    """
    device = get_device()
    dtype = get_dtype()

    print(f"Using device: {device}")
    if control_mode not in CONTROL_MODELS:
        raise ValueError(
            f"Unknown control mode: {control_mode}. "
            f"Expected one of: {', '.join(CONTROL_MODELS)}"
        )

    controlnet_model_id = CONTROL_MODELS[control_mode]

    print(f"Loading {control_mode} ControlNet: {controlnet_model_id}")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
    )

    print(f"Loading VAE: {VAE_MODEL_ID}")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=dtype,
    )

    print(f"Loading SDXL base model: {BASE_MODEL_ID}")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    # Helpful for VRAM stability
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception as exc:
            print(f"xFormers not enabled: {exc}")

    if lora_path:
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)

    return pipe


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

def generate_image(
    pipe,
    sketch: Image.Image,
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
    control_mode: str = DEFAULT_CONTROL_MODE,
) -> Tuple[Image.Image, Image.Image, str]:
    """
    Generates image from sketch + text prompt.

    Returns:
    generated_image, control_image, final_prompt
    """

    device = get_device()

    control_image = make_control_image(
        sketch,
        control_mode=control_mode,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        size=size,
    )

    final_prompt = build_prompt(prompt, lora_trigger=lora_trigger)

    generator = torch.Generator(device=device).manual_seed(int(seed))

    output = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        width=size,
        height=size,
    ).images[0]

    return output, control_image, final_prompt


# ------------------------------------------------------------
# CLI support
# ------------------------------------------------------------

def save_generation(
    pipe,
    sketch_path: str,
    prompt: str,
    output_path: str = "outputs/generated.png",
    control_path: str = "outputs/control_image.png",
    **kwargs,
):
    sketch = Image.open(sketch_path)

    image, control_image, final_prompt = generate_image(
        pipe=pipe,
        sketch=sketch,
        prompt=prompt,
        **kwargs,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(control_path).parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)
    control_image.save(control_path)

    print("Generation complete.")
    print(f"Final prompt: {final_prompt}")
    print(f"Generated image saved to: {output_path}")
    print(f"Control image saved to: {control_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SDXL dark fantasy character from sketch.")
    parser.add_argument("--sketch", default="examples/sample_sketches/sketch.png")
    parser.add_argument("--prompt", default="full body dark fantasy knight holding a glowing sword")
    parser.add_argument("--output", default="outputs/generated.png")
    parser.add_argument("--control-output", default="outputs/control_image.png")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--control-scale", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--low-threshold", type=int, default=80)
    parser.add_argument("--high-threshold", type=int, default=180)
    parser.add_argument("--control-mode", choices=sorted(CONTROL_MODELS), default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-trigger", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.sketch):
        raise FileNotFoundError(f"Sketch not found: {args.sketch}")

    pipe = load_pipeline(
        lora_path=args.lora_path or None,
        control_mode=args.control_mode,
    )

    save_generation(
        pipe=pipe,
        sketch_path=args.sketch,
        prompt=args.prompt,
        output_path=args.output,
        control_path=args.control_output,
        steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        seed=args.seed,
        size=args.size,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        control_mode=args.control_mode,
        lora_trigger=args.lora_trigger,
    )
