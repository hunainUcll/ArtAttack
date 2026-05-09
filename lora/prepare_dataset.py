"""Create the 300-image Art Attack LoRA training dataset."""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# Dataset source and local training output.
DATASET_NAME = "0xJustin/Dungeons-and-Diffusion"
LORA_DIR = Path(__file__).resolve().parent
TARGET_DIR = LORA_DIR / "train_dataset_300"
TARGET_IMAGES = TARGET_DIR / "images"
TARGET_METADATA = TARGET_DIR / "metadata.jsonl"

# Training captions all start with this trigger word.
TRIGGER_WORD = "artattackstyle"
TARGET_COUNT = 300
PER_RACE_TARGET = 10
RANDOM_SEED = 42

# Race labels used by the dataset captions.
RACES = [
    "aarakocra",
    "aasimar",
    "air genasi",
    "centaur",
    "dragonborn",
    "drow",
    "dwarf",
    "earth genasi",
    "elf",
    "firbolg",
    "fire genasi",
    "gith",
    "gnome",
    "goblin",
    "goliath",
    "halfling",
    "human",
    "illithid",
    "kenku",
    "kobold",
    "lizardfolk",
    "minotaur",
    "orc",
    "tabaxi",
    "thrikreen",
    "tiefling",
    "tortle",
    "warforged",
    "water genasi",
    "genasi",
]


# Find the race label inside a caption.
def detect_race(caption: str) -> str:
    normalized = caption.lower().replace("_", " ")
    for race in sorted(RACES, key=len, reverse=True):
        if f" {race} " in f" {normalized} ":
            return race
    return "unknown"


# Add the project trigger word and remove the repeated dataset prefix.
def clean_caption(caption: str) -> str:
    caption = caption.strip()
    if caption.lower().startswith("d&d character,"):
        caption = caption[len("d&d character,") :].strip()
    return f"{TRIGGER_WORD}, dark fantasy rpg character, {caption}"


# Download, balance, and save the small local dataset.
def main():
    random.seed(RANDOM_SEED)
    dataset = load_dataset(DATASET_NAME, split="train")

    rows = []
    for index, row in enumerate(dataset):
        caption = row["text"]
        rows.append(
            {
                "source_index": index,
                "image": row["image"],
                "text": clean_caption(caption),
                "race": detect_race(caption),
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["race"]].append(row)

    selected = []
    for race in sorted(grouped):
        if race == "unknown":
            continue
        random.shuffle(grouped[race])
        selected.extend(grouped[race][:PER_RACE_TARGET])

    selected_ids = {row["source_index"] for row in selected}
    remaining = [row for row in rows if row["source_index"] not in selected_ids]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, TARGET_COUNT - len(selected))])
    selected = selected[:TARGET_COUNT]

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_IMAGES.mkdir(parents=True, exist_ok=True)

    with TARGET_METADATA.open("w", encoding="utf-8") as metadata:
        for index, row in enumerate(selected, start=1):
            image_name = f"{index:05d}.png"
            row["image"].convert("RGB").save(TARGET_IMAGES / image_name)
            metadata.write(
                json.dumps(
                    {
                        "file_name": f"images/{image_name}",
                        "text": row["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    race_counts = defaultdict(int)
    for row in selected:
        race_counts[row["race"]] += 1

    print(f"Created {len(selected)} images in {TARGET_DIR}")
    for race, count in sorted(race_counts.items()):
        print(f"{race}: {count}")


# Run the dataset builder when this file is executed directly.
if __name__ == "__main__":
    main()
