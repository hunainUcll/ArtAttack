"""Shared configuration for Art Attack."""

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
CONTROL_MODES = ["none", *CONTROL_MODELS]
DEFAULT_CONTROL_MODE: Literal["none", "pose", "scribble", "canny"] = "pose"

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
        "full body dark fantasy knight, heroic wide battle stance, blackened steel plate armor, "
        "engraved shoulder guards, torn crimson cape, weathered leather belts, glowing runic sword, "
        "battle-worn helmet, ash and sparks in the air, ruined castle courtyard, realistic character concept art"
    ),
    "Necromancer": (
        "full body dark fantasy necromancer, tall thin silhouette, layered black ritual robes, "
        "bone ornaments and silver chains, skull staff raised in one hand, sickly green magic aura, "
        "floating embers and ghostly mist, ancient graveyard with cracked tombstones, realistic character concept art"
    ),
    "Rogue Assassin": (
        "full body dark fantasy rogue assassin, low agile stance, dark fitted leather armor, "
        "hooded cloak, wrapped forearms, twin curved daggers, hidden throwing knives, "
        "masked face with sharp eyes, rain-soaked shadowy alley, realistic character concept art"
    ),
    "Orc Warrior": (
        "full body dark fantasy orc warrior, massive muscular body, intimidating forward stance, "
        "scarred green skin, heavy spiked iron armor, fur-lined pauldrons, massive chipped battle axe, "
        "war paint, broken chains, smoky battlefield background, realistic character concept art"
    ),
    "Demon Hunter": (
        "full body dark fantasy demon hunter, confident monster-slayer pose, long worn black coat, "
        "cursed silver armor pieces, glowing red blade, holy symbols and potion vials on belt, "
        "burn scars, determined expression, hellish battlefield with demonic smoke, realistic character concept art"
    ),
    "Undead King": (
        "full body undead king, regal threatening stance, ancient tarnished royal armor, "
        "crown made of bone and black iron, glowing blue eyes, skeletal hands, tattered royal cloak, "
        "frosty breath, cracked throne, ruined gothic throne room, realistic character concept art"
    ),
    "Forest Witch": (
        "full body dark fantasy forest witch, mysterious upright pose, layered black and moss-green dress, "
        "twisted wooden staff with glowing crystals, wild silver hair, bark and thorn details, "
        "green magical aura, floating leaves, haunted moonlit forest background, realistic character concept art"
    ),
}
