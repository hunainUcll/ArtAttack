"""Configuration for Sketch2DarkFantasy."""

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, bad anatomy, bad hands, "
    "extra fingers, missing fingers, extra limbs, distorted face, "
    "deformed body, ugly, text, watermark, logo, cropped, duplicate body"
)

STYLE_SUFFIX = (
    "dark fantasy character concept art, highly detailed, realistic, "
    "cinematic lighting, dramatic atmosphere, full body, sharp focus, "
    "detailed armor, professional concept art"
)

PRESETS = {
    "Dark Fantasy Knight": "armored knight, black steel armor, glowing sword, ruined castle background",
    "Necromancer": "necromancer, skull staff, black robes, green magical aura, graveyard background",
    "Rogue Assassin": "rogue assassin, leather armor, daggers, hooded cloak, moonlit alley",
    "Orc Warrior": "orc warrior, heavy armor, giant axe, battle scars, smoky battlefield",
    "Demon Hunter": "demon hunter, long coat, glowing runes, silver sword, hellish background",
    "Forest Witch": "forest witch, antler crown, dark robes, magical staff, haunted forest",
    "Undead King": "undead king, ancient crown, bone armor, throne room, blue ghostly flame",
}
