"""
Gradio interface for Sketch2DarkFantasy.

Run with:
    python app.py
"""

import gradio as gr

from generate import generate_image, load_pipeline
from src.config import DEFAULT_NEGATIVE_PROMPT, PRESETS

PIPE = None


def get_pipe(lora_path: str = ""):
    global PIPE
    if PIPE is None:
        PIPE = load_pipeline(lora_path=lora_path.strip() or None)
    return PIPE


def apply_preset(preset_name: str, current_prompt: str):
    preset_prompt = PRESETS.get(preset_name, "")
    if current_prompt.strip():
        return f"{current_prompt.strip()}, {preset_prompt}"
    return preset_prompt


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
    lora_trigger,
):
    if sketch is None:
        raise gr.Error("Please upload or draw a sketch first.")

    if not prompt.strip():
        prompt = PRESETS.get(preset_name, "a dark fantasy character")

    pipe = get_pipe()
    generated, control_image, final_prompt = generate_image(
        pipe=pipe,
        sketch=sketch,
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        lora_trigger=lora_trigger,
    )
    settings = (
        f"Final prompt: {final_prompt}\n\n"
        f"Steps: {steps}\n"
        f"Guidance scale: {guidance_scale}\n"
        f"ControlNet scale: {controlnet_conditioning_scale}\n"
        f"Seed: {seed}\n"
        f"Canny thresholds: {low_threshold}, {high_threshold}"
    )
    return generated, control_image, settings


with gr.Blocks(title="Sketch2DarkFantasy") as demo:
    gr.Markdown(
        "# Sketch2DarkFantasy\n"
        "Generate dark fantasy character concept art from a rough sketch and text prompt using Stable Diffusion + ControlNet."
    )

    with gr.Row():
        with gr.Column():
            sketch_input = gr.Image(label="Upload or draw sketch", type="pil", image_mode="RGB")
            preset = gr.Dropdown(choices=list(PRESETS.keys()), value="Dark Fantasy Knight", label="Character preset")
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Example: hooded knight holding a glowing sword")
            preset_button = gr.Button("Add preset to prompt")
            negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=3)

        with gr.Column():
            output_image = gr.Image(label="Generated image")
            control_image = gr.Image(label="ControlNet Canny image")
            settings_output = gr.Textbox(label="Generation settings", lines=8)

    with gr.Accordion("Advanced settings", open=False):
        steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")
        guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
        controlnet_conditioning_scale = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="ControlNet conditioning scale")
        seed = gr.Number(value=42, precision=0, label="Seed")
        low_threshold = gr.Slider(1, 255, value=100, step=1, label="Canny low threshold")
        high_threshold = gr.Slider(1, 255, value=200, step=1, label="Canny high threshold")
        lora_trigger = gr.Textbox(label="Optional LoRA trigger word", placeholder="Example: dfcharstyle")

    generate_button = gr.Button("Generate", variant="primary")

    preset_button.click(apply_preset, inputs=[preset, prompt], outputs=prompt)
    generate_button.click(
        run_generation,
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
            lora_trigger,
        ],
        outputs=[output_image, control_image, settings_output],
    )


if __name__ == "__main__":
    demo.launch()
