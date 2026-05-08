"""Gradio app for Sketch2DarkFantasy."""

import os

import gradio as gr

from generate import generate_image, load_pipeline
from src.config import (
    BASE_MODEL_ID,
    CONTROL_MODELS,
    DEFAULT_CONTROL_MODE,
    DEFAULT_NEGATIVE_PROMPT,
    PROMPT_PRESETS,
)

PIPE = None
CACHE_KEY = None


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def get_pipe(lora_path: str = "", control_mode: str = DEFAULT_CONTROL_MODE):
    global PIPE, CACHE_KEY

    lora_path = clean_text(lora_path)
    control_mode = control_mode if control_mode in CONTROL_MODELS else DEFAULT_CONTROL_MODE
    cache_key = (lora_path, control_mode)

    if PIPE is None or CACHE_KEY != cache_key:
        PIPE = load_pipeline(lora_path=lora_path or None, control_mode=control_mode)
        CACHE_KEY = cache_key

    return PIPE


def apply_preset(preset_name: str):
    return PROMPT_PRESETS.get(preset_name, "")


def settings_markdown(values: dict) -> str:
    return f"""
## Generation Settings

**Base model:** `{BASE_MODEL_ID}`  
**Conditioning mode:** `{values["control_mode"]}`  
**Conditioning model:** `{CONTROL_MODELS[values["control_mode"]]}`  
**Preset:** `{values["preset_name"]}`  
**Image size:** `{values["image_size"]}x{values["image_size"]}`  
**Steps:** `{values["steps"]}`  
**Guidance scale:** `{values["guidance_scale"]}`  
**Conditioning scale:** `{values["controlnet_conditioning_scale"]}`  
**Canny low threshold:** `{values["low_threshold"]}`  
**Canny high threshold:** `{values["high_threshold"]}`  
**Seed:** `{values["seed"]}`  
**LoRA path:** `{values["lora_path"] or "None"}`  
**LoRA trigger:** `{values["lora_trigger"] or "None"}`  

### Final Prompt

`{values["final_prompt"]}`

### Negative Prompt

`{values["negative_prompt"]}`
"""


def run_generation(
    sketch,
    preset_name,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    controlnet_conditioning_scale,
    seed,
    low_threshold,
    high_threshold,
    lora_path,
    lora_trigger,
    image_size,
    control_mode,
):
    if sketch is None:
        raise gr.Error("Please upload or draw a sketch first.")

    prompt = clean_text(prompt) or PROMPT_PRESETS.get(preset_name, "")
    if not prompt:
        raise gr.Error("Please enter a prompt or choose a preset.")

    negative_prompt = clean_text(negative_prompt) or DEFAULT_NEGATIVE_PROMPT
    lora_path = clean_text(lora_path)
    lora_trigger = clean_text(lora_trigger)

    generated, control_image, final_prompt = generate_image(
        pipe=get_pipe(lora_path=lora_path, control_mode=control_mode),
        sketch=sketch,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        seed=int(seed),
        size=int(image_size),
        low_threshold=int(low_threshold),
        high_threshold=int(high_threshold),
        lora_trigger=lora_trigger,
        control_mode=control_mode,
    )

    return generated, control_image, settings_markdown(locals())


with gr.Blocks(title="Sketch2DarkFantasy") as demo:
    gr.Markdown(
        """
# Sketch2DarkFantasy

Generate dark fantasy character concept art from a rough sketch and a text prompt.

This prototype uses:

`Stable Diffusion XL + Pose/Scribble/Canny conditioning + optional LoRA`

Pose mode is best for stick-figure body poses. Scribble mode is better for rough shape sketches.  
The prompt controls the character details.  
The optional LoRA controls the art style.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketch_input = gr.Image(label="Input Sketch", type="pil", image_mode="RGB", sources=["upload", "clipboard"])
            preset = gr.Dropdown(choices=list(PROMPT_PRESETS), value="Dark Fantasy Knight", label="Prompt preset")
            prompt = gr.Textbox(
                label="Prompt",
                value=PROMPT_PRESETS["Dark Fantasy Knight"],
                lines=4,
                placeholder="Example: full body dark fantasy knight holding a glowing sword",
            )
            negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=4)
            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")
            control_image = gr.Image(label="Conditioning Image", type="pil")

    with gr.Accordion("Advanced Settings", open=True):
        with gr.Row():
            image_size = gr.Dropdown(choices=[768, 1024], value=1024, label="Image size")
            control_mode = gr.Dropdown(choices=list(CONTROL_MODELS), value=DEFAULT_CONTROL_MODE, label="Sketch conditioning")
            steps = gr.Slider(minimum=20, maximum=60, value=35, step=1, label="Inference steps")

        with gr.Row():
            guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="Guidance scale")
            controlnet_conditioning_scale = gr.Slider(minimum=0.2, maximum=1.5, value=0.9, step=0.05, label="Conditioning scale")

        with gr.Row():
            low_threshold = gr.Slider(minimum=1, maximum=255, value=80, step=1, label="Canny low threshold")
            high_threshold = gr.Slider(minimum=1, maximum=255, value=180, step=1, label="Canny high threshold")

        seed = gr.Number(value=1234, precision=0, label="Seed")

    with gr.Accordion("LoRA Settings", open=False):
        gr.Markdown(
            """
Use this only after you have trained or downloaded a compatible **SDXL LoRA**.

Example trigger word:

`artattackstyle`
"""
        )
        lora_path = gr.Textbox(
            label="LoRA path",
            value="",
            placeholder="Example: C:/Users/Admin/Documents/loras/artattackstyle.safetensors",
        )
        lora_trigger = gr.Textbox(label="LoRA trigger word", value="", placeholder="Example: artattackstyle")

    settings_output = gr.Markdown(label="Settings")

    preset.change(fn=apply_preset, inputs=preset, outputs=prompt)
    generate_button.click(
        fn=run_generation,
        inputs=[
            sketch_input,
            preset,
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            low_threshold,
            high_threshold,
            lora_path,
            lora_trigger,
            image_size,
            control_mode,
        ],
        outputs=[output_image, control_image, settings_output],
    )


if __name__ == "__main__":
    demo.queue()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
