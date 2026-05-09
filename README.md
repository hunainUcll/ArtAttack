# Sketch2DarkFantasy

**Sketch2DarkFantasy** is a semester project prototype that turns a rough user sketch and a short text prompt into dark fantasy character concept art.

The project uses:

- Stable Diffusion for image generation
- ControlNet for pose and sketch conditioning
- Pose preprocessing for stick figures, with Scribble and Canny fallbacks
- Gradio for the user interface
- Optional LoRA support for dark fantasy style adaptation

## Project goal

Many people have creative ideas for fantasy characters but cannot draw them. Text-to-image tools can create impressive images, but the user has limited control over pose and composition. This project solves that by combining two inputs:

1. A rough sketch that controls pose/composition
2. A text prompt that controls character details/style

## Pipeline

```text
User sketch + text prompt
        ↓
Image preprocessing with Pose, Scribble, or Canny conditioning
        ↓
ControlNet conditioning
        ↓
Stable Diffusion generation
        ↓
Dark fantasy character output
```

## Installation

A GPU environment such as Google Colab is strongly recommended.

```bash
git clone <your-repo-url>
cd sketch2darkfantasy
pip install -r requirements.txt
```

## Run the Gradio app

```bash
python app.py
```

Then open the local Gradio link, upload or draw a sketch, write a prompt, and click **Generate**.

## Run from command line

Put a sketch at:

```text
examples/sample_sketches/sketch.png
```

Then run:

```bash
python generate.py --sketch examples/sample_sketches/sketch.png --prompt "hooded knight holding a glowing sword"
```

The generated image will be saved to:

```text
outputs/generated.png
```

The conditioning image will be saved to:

```text
outputs/control_image.png
```

## Example prompts

```text
hooded knight holding a glowing sword, black armor, ancient ruins
```

```text
necromancer with skull staff, black robes, green magical aura, graveyard background
```

```text
orc warrior with giant axe, heavy armor, smoky battlefield, dramatic lighting
```

## Important parameters

| Parameter | Meaning |
|---|---|
| Inference steps | Number of denoising steps. Higher can improve quality but takes longer. |
| Guidance scale | How strongly the model follows the text prompt. |
| Conditioning scale | How strongly the model follows the sketch/line map. |
| Seed | Makes results reproducible. |
| Sketch conditioning | `pose` is best for stick-figure body poses; `scribble` is useful for rough shape sketches; `canny` is available for cleaner edge sketches. |
| Canny thresholds | Control how much detail is extracted from the sketch when Canny mode is selected. |

## Optional LoRA extension

The project supports the idea of adding a LoRA trigger word and loading LoRA weights. A LoRA can make the output more consistent with a dark fantasy style.

In the final defense, describe this honestly:

> The main working prototype uses pretrained Stable Diffusion and ControlNet. LoRA adaptation is included as an extension to specialize the output style toward dark fantasy character art.

## Limitations

- Very rough sketches may not give enough structure for the model.
- Hands, weapons, and faces can become distorted.
- The model follows clear silhouettes better than messy sketches.
- A custom LoRA could improve style consistency but requires a clean dataset and GPU training time.

## Future improvements

- Add a browser drawing canvas.
- Train a small custom LoRA on dark fantasy character art.
- Add side-by-side comparison with and without LoRA.
- Improve OpenPose-style pose preprocessing for messy sketches and props.
- Add automatic prompt suggestions.
  