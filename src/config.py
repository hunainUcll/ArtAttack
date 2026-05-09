"""Shared configuration for Sketch2DarkFantasy."""

from typing import Literal

# Model ids used by the SDXL generation pipeline.
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

# ControlNet choices shown in the app and CLI.
CONTROL_MODELS = {
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",
    "scribble": "xinsir/controlnet-scribble-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
}
DEFAULT_CONTROL_MODE: Literal["pose", "scribble", "canny"] = "pose"

# Text that tells the model what to avoid.
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, pixelated, bad anatomy, bad hands, "
    "extra fingers, missing fingers, extra limbs, missing limbs, deformed body, "
    "distorted face, ugly face, cropped, out of frame, text, watermark, logo, "
    "cartoon, childish, stick figure, doodle, simple line art"
)

# Style text added to every prompt.
STYLE_SUFFIX = (
    "dark fantasy character concept art, full body character, highly detailed, "
    "realistic, cinematic lighting, dramatic atmosphere, sharp focus, "
    "professional fantasy illustration, detailed armor, dark medieval background"
)

# Ready-made prompts for the dropdown in the app.
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
