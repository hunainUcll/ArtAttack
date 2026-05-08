"""Pony Diffusion XL + ControlNet generation for Sketch2DarkFantasy."""

import argparse
import inspect
import os
from pathlib import Path
from typing import Tuple

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

from src.config import (
    BASE_MODEL_ID,
    BASE_MODEL_VARIANT,
    CLIP_SKIP,
    CONTROL_MODELS,
    DEFAULT_CONTROL_MODE,
    DEFAULT_NEGATIVE_PROMPT,
    PONY_PROMPT_PREFIX,
    STYLE_SUFFIX,
    VAE_MODEL_ID,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    return torch.float16 if get_device() == "cuda" else torch.float32


def resize_image(image: Image.Image, size: int = 1024) -> Image.Image:
    return image.convert("RGB").resize((size, size))


def make_canny_image(
    image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    image = resize_image(image, size=size)
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


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


def make_pose_image(
    image: Image.Image,
    size: int = 1024,
    threshold: int = 210,
) -> Image.Image:
    """Approximate an OpenPose-style control map from a simple stick figure."""
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


def make_control_image(
    image: Image.Image,
    control_mode: str = DEFAULT_CONTROL_MODE,
    low_threshold: int = 80,
    high_threshold: int = 180,
    size: int = 1024,
) -> Image.Image:
    if control_mode == "canny":
        return make_canny_image(image, low_threshold, high_threshold, size)

    if control_mode == "scribble":
        return make_scribble_image(image, size=size)

    if control_mode == "pose":
        return make_pose_image(image, size=size)

    raise ValueError(
        f"Unknown control mode: {control_mode}. "
        f"Expected one of: {', '.join(CONTROL_MODELS)}"
    )


def build_prompt(prompt: str, lora_trigger: str = "") -> str:
    parts = [
        part.strip()
        for part in (
            PONY_PROMPT_PREFIX,
            lora_trigger or "",
            prompt or "",
            STYLE_SUFFIX,
        )
    ]
    return ", ".join(part for part in parts if part)

def load_pipeline(
    lora_path: str | None = None,
    control_mode: str = DEFAULT_CONTROL_MODE,
):
    device = get_device()
    dtype = get_dtype()

    if control_mode not in CONTROL_MODELS:
        raise ValueError(
            f"Unknown control mode: {control_mode}. "
            f"Expected one of: {', '.join(CONTROL_MODELS)}"
        )

    print(f"Using device: {device}")
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

    print(f"Loading Pony Diffusion XL base model: {BASE_MODEL_ID}")
    model_kwargs = {
        "controlnet": controlnet,
        "vae": vae,
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if BASE_MODEL_VARIANT:
        model_kwargs["variant"] = BASE_MODEL_VARIANT

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        **model_kwargs,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

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

    generation_kwargs = {
        "prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "image": control_image,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "generator": generator,
        "width": size,
        "height": size,
    }

    if "clip_skip" in inspect.signature(pipe.__call__).parameters:
        generation_kwargs["clip_skip"] = CLIP_SKIP

    output = pipe(**generation_kwargs).images[0]

    return output, control_image, final_prompt


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
    parser = argparse.ArgumentParser(description="Generate Pony Diffusion XL dark fantasy character from sketch.")
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
