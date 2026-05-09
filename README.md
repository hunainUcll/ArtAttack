# Art Attack

Art Attack is a semester project prototype that turns a rough sketch and a text prompt into dark fantasy character concept art.

The project uses Stable Diffusion XL for image generation, optional ControlNet conditioning for sketches, and an experimental SDXL LoRA trained on fantasy RPG character art.

## Features 

- Gradio interface in `app.py`
- Prompt-only generation with `none` mode
- Sketch-conditioned generation with `pose`, `scribble`, and `canny`
- Optional LoRA loading from `lora/trained_lora`
- Latest generated files saved to `results/input_1.png` and `results/output_1.png`

## Project Structure

```text
app.py                  Gradio user interface
generate.py             SDXL, ControlNet, LoRA, and result-saving logic
src/config.py           Model IDs, presets, prompts, and settings
lora/                   Dataset preparation notebook/script and trained LoRA
results/                Latest generated input/output images
docs/vent.txt           Development notes and rough decision log
technical_report.md     Final project report
```

## Setup

```bash
pip install -r requirements.txt
```

An NVIDIA GPU is strongly recommended. The project was tested with an RTX 4070 Ti SUPER.

## Run The App

```bash
python app.py
```

Open the local Gradio URL, usually:

```text
http://127.0.0.1:7860
```

## Suggested Test Settings

For a fast LoRA test:

- Sketch conditioning: `none`
- LoRA path: `lora/trained_lora`
- LoRA trigger: `artattackstyle`
- LoRA scale: `0.6`
- Image size: `768`
- Steps: `25`

Example prompt:

```text
full body dark fantasy tiefling warrior, black armor, glowing sword, ruined castle background
```

## LoRA Training

The quick local LoRA was trained with:

- Dataset: `0xJustin/Dungeons-and-Diffusion`
- Local subset: `lora/train_dataset_300`
- Images: 300
- Resolution: 512
- Steps: 300
- Rank: 8
- Trigger word: `artattackstyle`

To recreate the 300-image subset:

```bash
python lora/prepare_dataset.py
```

To train from the notebook, open:

```text
lora/train_art_attack_lora.ipynb
```

## Notes

The quick LoRA is only an experiment. Because it was trained for speed, it may produce subtle changes at low scale and artifacts at high scale. Start around `0.4` to `0.6`, then increase carefully.
