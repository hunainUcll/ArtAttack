"""
Sketch2DarkFantasy - Gradio Web App

This app lets the user upload a rough sketch and generate a dark fantasy
character image using SDXL + ControlNet.
"""

import gradio as gr

from generate import (
    load_pipeline,
    generate_image,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_CONTROL_MODE,
    CONTROL_MODELS,
)


# ------------------------------------------------------------
# Global pipeline cache
# ------------------------------------------------------------

PIPE = None
CURRENT_LORA_PATH = None
CURRENT_CONTROL_MODE = None


def get_pipe(lora_path: str = "", control_mode: str = DEFAULT_CONTROL_MODE):
    """
    Loads the SDXL ControlNet pipeline once and reuses it.
    If a new LoRA path is provided, the pipeline reloads.
    """
    global PIPE, CURRENT_LORA_PATH, CURRENT_CONTROL_MODE

    if lora_path is None:
        lora_path = ""

    lora_path = lora_path.strip()

    if control_mode not in CONTROL_MODELS:
        control_mode = DEFAULT_CONTROL_MODE

    if (
        PIPE is None
        or CURRENT_LORA_PATH != lora_path
        or CURRENT_CONTROL_MODE != control_mode
    ):
        PIPE = load_pipeline(
            lora_path=lora_path if lora_path else None,
            control_mode=control_mode,
        )
        CURRENT_LORA_PATH = lora_path
        CURRENT_CONTROL_MODE = control_mode

    return PIPE


# ------------------------------------------------------------
# Prompt presets
# ------------------------------------------------------------

PROMPT_PRESETS = {
    "Custom": "",
    "Dark Fantasy Knight": (
        "full body dark fantasy knight, standing in a wide battle stance, "
        "black steel armor, long torn cape, glowing sword, realistic character concept art"
    ),
    "Necromancer": (
        "full body dark fantasy necromancer, black robes, skull staff, glowing green magic, "
        "ancient graveyard, realistic character concept art"
    ),
    "Rogue Assassin": (
        "full body dark fantasy rogue assassin, leather armor, hooded cloak, twin daggers, "
        "shadowy alley, realistic character concept art"
    ),
    "Orc Warrior": (
        "full body dark fantasy orc warrior, massive axe, heavy armor, battle scars, "
        "brutal medieval fantasy style, realistic character concept art"
    ),
    "Demon Hunter": (
        "full body dark fantasy demon hunter, long coat, glowing weapon, cursed armor, "
        "hellish background, realistic character concept art"
    ),
    "Undead King": (
        "full body undead king, dark fantasy armor, crown of bones, glowing eyes, "
        "ancient ruined throne room, realistic character concept art"
    ),
    "Forest Witch": (
        "full body dark fantasy forest witch, twisted wooden staff, black dress, magical aura, "
        "haunted forest background, realistic character concept art"
    ),
}


def apply_preset(preset_name: str):
    """
    Updates the prompt textbox when a preset is selected.
    """
    return PROMPT_PRESETS.get(preset_name, "")


# ------------------------------------------------------------
# Main generation function
# ------------------------------------------------------------

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
    """
    Called by the Gradio button.

    Takes the sketch and settings from the UI, generates an image,
    and returns:
    - generated image
    - ControlNet conditioning image
    - settings text
    """

    if sketch is None:
        raise gr.Error("Please upload or draw a sketch first.")

    if prompt is None or not prompt.strip():
        prompt = PROMPT_PRESETS.get(preset_name, "")

    if prompt is None or not prompt.strip():
        raise gr.Error("Please enter a prompt or choose a preset.")

    if negative_prompt is None or not negative_prompt.strip():
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if lora_path is None:
        lora_path = ""

    if lora_trigger is None:
        lora_trigger = ""

    pipe = get_pipe(lora_path=lora_path, control_mode=control_mode)

    generated, control_image, final_prompt = generate_image(
        pipe=pipe,
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

    settings_text = f"""
## Generation Settings

**Base model:** `stabilityai/stable-diffusion-xl-base-1.0`  
**Conditioning mode:** `{control_mode}`  
**Conditioning model:** `{CONTROL_MODELS[control_mode]}`  
**Preset:** `{preset_name}`  
**Image size:** `{image_size}x{image_size}`  
**Steps:** `{steps}`  
**Guidance scale:** `{guidance_scale}`  
**Conditioning scale:** `{controlnet_conditioning_scale}`  
**Canny low threshold:** `{low_threshold}`  
**Canny high threshold:** `{high_threshold}`  
**Seed:** `{seed}`  
**LoRA path:** `{lora_path if lora_path else "None"}`  
**LoRA trigger:** `{lora_trigger if lora_trigger else "None"}`  

### Final Prompt

`{final_prompt}`

### Negative Prompt

`{negative_prompt}`
"""

    return generated, control_image, settings_text


# ------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------

with gr.Blocks(title="Sketch2DarkFantasy") as demo:
    gr.Markdown(
        """
# Sketch2DarkFantasy

Generate dark fantasy character concept art from a rough sketch and a text prompt.

This prototype uses:

`Stable Diffusion XL + Scribble/Canny conditioning + optional LoRA`

The sketch controls the structure. Scribble mode is best for stick figures.  
The prompt controls the character details.  
The optional LoRA controls the art style.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketch_input = gr.Image(
                label="Input Sketch",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
            )

            preset = gr.Dropdown(
                choices=list(PROMPT_PRESETS.keys()),
                value="Dark Fantasy Knight",
                label="Prompt preset",
            )

            prompt = gr.Textbox(
                label="Prompt",
                value=PROMPT_PRESETS["Dark Fantasy Knight"],
                lines=4,
                placeholder="Example: full body dark fantasy knight holding a glowing sword",
            )

            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=4,
            )

            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
            )

            control_image = gr.Image(
                label="Conditioning Image",
                type="pil",
            )

    with gr.Accordion("Advanced Settings", open=True):
        with gr.Row():
            image_size = gr.Dropdown(
                choices=[768, 1024],
                value=1024,
                label="Image size",
            )

            control_mode = gr.Dropdown(
                choices=list(CONTROL_MODELS.keys()),
                value=DEFAULT_CONTROL_MODE,
                label="Sketch conditioning",
            )

            steps = gr.Slider(
                minimum=20,
                maximum=60,
                value=35,
                step=1,
                label="Inference steps",
            )

        with gr.Row():
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=7.0,
                step=0.5,
                label="Guidance scale",
            )

            controlnet_conditioning_scale = gr.Slider(
                minimum=0.2,
                maximum=1.5,
                value=0.9,
                step=0.05,
                label="Conditioning scale",
            )

        with gr.Row():
            low_threshold = gr.Slider(
                minimum=1,
                maximum=255,
                value=80,
                step=1,
                label="Canny low threshold",
            )

            high_threshold = gr.Slider(
                minimum=1,
                maximum=255,
                value=180,
                step=1,
                label="Canny high threshold",
            )

        seed = gr.Number(
            value=1234,
            precision=0,
            label="Seed",
        )

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

        lora_trigger = gr.Textbox(
            label="LoRA trigger word",
            value="",
            placeholder="Example: artattackstyle",
        )

    settings_output = gr.Markdown(label="Settings")

    preset.change(
        fn=apply_preset,
        inputs=preset,
        outputs=prompt,
    )

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
        outputs=[
            output_image,
            control_image,
            settings_output,
        ],
    )


# ------------------------------------------------------------
# Launch app
# ------------------------------------------------------------

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
