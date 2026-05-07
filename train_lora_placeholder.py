"""
LoRA training placeholder / documentation script.

For the final project, the main working component is the ControlNet-based generation
pipeline. LoRA training can be added as an extension using Hugging Face Diffusers'
official training examples.

Recommended dataset structure:

data/dark_fantasy_lora/
  images/
    0001.png
    0002.png
  metadata.jsonl

Each metadata line example:
{"file_name":"0001.png","text":"dfcharstyle, dark fantasy knight, black armor, glowing sword"}

Training a LoRA from scratch is GPU-heavy. For a student prototype, it is acceptable
to compare the base pipeline with an existing or small custom LoRA and explain the
limitations honestly.
"""

print(__doc__)
