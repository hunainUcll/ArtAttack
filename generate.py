"""
Sketch2DarkFantasy generation module.

This file contains the core AI pipeline:
user sketch -> Canny preprocessing -> ControlNet -> Stable Diffusion -> output image.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

from src.config import BASE_MODEL_ID, CONTROLNET_MODEL_ID, DEFAULT_NEGATIVE_PROMPT, STYLE_SUFFIX


def get_device() -> str:
    """Return cuda when available, otherwise cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resize_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Convert image to RGB and resize it to a square image."""
    return image.convert("RGB").resize((size, size))


def make_canny_image(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    size: int = 512,
) -> Image.Image:
    """
    Convert an input sketch into a Canny edge map.

    ControlNet-Canny expects a 3-channel edge image. Even when the original image is
    already a sketch, Canny makes the conditioning format more consistent.
    """
    image = resize_image(image, size=size)
    image_np = np.array(image)
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def build_prompt(user_prompt: str, use_style_suffix: bool = True, lora_trigger: str = "") -> str:
    """Combine user prompt with the dark fantasy style suffix and optional LoRA trigger word."""
    parts = []
    if lora_trigger.strip():
        parts.append(lora_trigger.strip())
    parts.append(user_prompt.strip())
    if use_style_suffix:
        parts.append(STYLE_SUFFIX)
    return ", ".join([part for part in parts if part])


def load_pipeline(
    base_model_id: str = BASE_MODEL_ID,
    controlnet_model_id: str = CONTROLNET_MODEL_ID,
    lora_path: Optional[str] = None,
):
    """Load Stable Diffusion with ControlNet and optionally load a LoRA adapter."""
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading ControlNet: {controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype)

    print(f"Loading Stable Diffusion base model: {base_model_id}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory efficient attention.")
        except Exception as exc:
            print(f"xFormers not enabled: {exc}")

    if lora_path: # i will hopefully have enough time to immplement a personalized lora style
        print(f"Loading LoRA adapter from: {lora_path}")
        pipe.load_lora_weights(lora_path)

    return pipe


def generate_image(
    pipe,
    sketch: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    seed: int = 42,
    size: int = 512,
    low_threshold: int = 100,
    high_threshold: int = 200,
    lora_trigger: str = "",
) -> Tuple[Image.Image, Image.Image, str]:
    """
    Generate an image from a sketch and prompt.

    Returns:
        generated_image, control_image, final_prompt
    """
    device = get_device()
    control_image = make_canny_image(sketch, low_threshold, high_threshold, size=size)
    final_prompt = build_prompt(prompt, use_style_suffix=True, lora_trigger=lora_trigger)

    generator = torch.Generator(device=device).manual_seed(int(seed))

    output = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
    ).images[0]

    return output, control_image, final_prompt


def save_generation(
    pipe,
    sketch_path: str,
    prompt: str,
    output_path: str = "outputs/generated.png",
    control_path: str = "outputs/control_image.png",
    **kwargs,
) -> None:
    """Generate an image from a sketch file and save both output and control image."""
    sketch = Image.open(sketch_path)
    image, control_image, final_prompt = generate_image(pipe, sketch, prompt, **kwargs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(control_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    control_image.save(control_path)

    print("Generation finished.")
    print(f"Final prompt: {final_prompt}")
    print(f"Generated image saved to: {output_path}")
    print(f"Control image saved to: {control_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dark fantasy character art from a sketch.")
    parser.add_argument("--sketch", default="examples/sample_sketches/sketch.png", help="Path to input sketch image")
    parser.add_argument("--prompt", default="a hooded knight holding a glowing sword", help="Text prompt")
    parser.add_argument("--output", default="outputs/generated.png", help="Output image path")
    parser.add_argument("--control-output", default="outputs/control_image.png", help="Control image output path")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-path", default="", help="Optional path or repo id for LoRA weights")
    parser.add_argument("--lora-trigger", default="", help="Optional LoRA trigger word")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.sketch):
        raise FileNotFoundError(
            f"Sketch not found: {args.sketch}\n"
            "Put a sketch image there or pass --sketch path/to/image.png"
        )

    pipe = load_pipeline(lora_path=args.lora_path or None)
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
        lora_trigger=args.lora_trigger,
    )
