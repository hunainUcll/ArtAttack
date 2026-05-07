"""
Sketch2DarkFantasy generation module.

Pipeline:
user sketch -> pose/edge preprocessing -> ControlNet -> Stable Diffusion -> output image.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from controlnet_aux import OpenposeDetector

from src.config import BASE_MODEL_ID, DEFAULT_NEGATIVE_PROMPT, STYLE_SUFFIX


CANNY_CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
OPENPOSE_CONTROLNET_ID = "lllyasviel/sd-controlnet-openpose"

OPENPOSE_PROCESSOR = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resize_image(image: Image.Image, size: int = 512) -> Image.Image:
    return image.convert("RGB").resize((size, size))


def make_canny_image(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    size: int = 512,
) -> Image.Image:
    image = resize_image(image, size=size)
    image_np = np.array(image)
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def make_openpose_image(image: Image.Image, size: int = 512) -> Image.Image:
    """
    Creates an OpenPose conditioning image.

    This works best when the input is a rough humanoid figure.
    It gives ControlNet body-pose guidance instead of raw edge guidance.
    """
    global OPENPOSE_PROCESSOR

    image = resize_image(image, size=size)

    if OPENPOSE_PROCESSOR is None:
        print("Loading OpenPose processor...")
        OPENPOSE_PROCESSOR = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    pose_image = OPENPOSE_PROCESSOR(image)
    pose_image = resize_image(pose_image, size=size)
    return pose_image


def build_prompt(user_prompt: str, use_style_suffix: bool = True, lora_trigger: str = "") -> str:
    if user_prompt is None:
        user_prompt = ""

    if lora_trigger is None:
        lora_trigger = ""

    user_prompt = user_prompt.strip()
    lora_trigger = lora_trigger.strip()

    parts = []

    if lora_trigger:
        parts.append(lora_trigger)

    if user_prompt:
        parts.append(user_prompt)

    if use_style_suffix:
        parts.append(STYLE_SUFFIX)

    return ", ".join(parts)


def load_pipeline(
    base_model_id: str = BASE_MODEL_ID,
    control_mode: str = "openpose",
    lora_path: Optional[str] = None,
):
    """
    Load Stable Diffusion with either OpenPose ControlNet or Canny ControlNet.
    """
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    control_mode = (control_mode or "openpose").lower().strip()

    if control_mode == "canny":
        controlnet_model_id = CANNY_CONTROLNET_ID
    else:
        controlnet_model_id = OPENPOSE_CONTROLNET_ID

    print(f"Using device: {device}")
    print(f"Loading ControlNet mode: {control_mode}")
    print(f"Loading ControlNet model: {controlnet_model_id}")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
    )

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

    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        pipe.load_lora_weights(lora_path)

    return pipe


def generate_image(
    pipe,
    sketch: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 35,
    guidance_scale: float = 8.5,
    controlnet_conditioning_scale: float = 0.85,
    seed: int = 42,
    size: int = 512,
    low_threshold: int = 100,
    high_threshold: int = 200,
    lora_trigger: str = "",
    control_mode: str = "openpose",
) -> Tuple[Image.Image, Image.Image, str]:
    """
    Generate an image from a sketch and prompt.

    Returns:
        generated_image, control_image, final_prompt
    """
    device = get_device()
    control_mode = (control_mode or "openpose").lower().strip()

    if control_mode == "canny":
        control_image = make_canny_image(
            sketch,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            size=size,
        )
    else:
        control_image = make_openpose_image(sketch, size=size)

    final_prompt = build_prompt(
        prompt,
        use_style_suffix=True,
        lora_trigger=lora_trigger,
    )

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
    parser.add_argument("--sketch", default="examples/sample_sketches/sketch.png")
    parser.add_argument("--prompt", default="full body dark fantasy knight holding a glowing sword")
    parser.add_argument("--output", default="outputs/generated.png")
    parser.add_argument("--control-output", default="outputs/control_image.png")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=8.5)
    parser.add_argument("--control-scale", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-trigger", default="")
    parser.add_argument("--control-mode", default="openpose", choices=["openpose", "canny"])
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
        lora_trigger=args.lora_trigger,
        control_mode=args.control_mode,
    )