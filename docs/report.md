# Sketch2DarkFantasy Report

## 1. Introduction

The goal of this project is to create a prototype that generates dark fantasy character concept art from a rough user sketch and a short text prompt. The target users are people who have a creative idea for a character but do not have strong drawing skills.

Pure text-to-image systems can generate impressive images, but the user has limited control over pose and composition. This project improves user control by allowing the sketch to define the rough pose and silhouette, while the text prompt defines the character identity, clothing, weapon, environment, and style.

## 2. Project idea

The application takes two inputs:

1. A rough sketch or stick-figure drawing
2. A short text description

The system then generates a detailed dark fantasy character image using Stable Diffusion with ControlNet.

Example input prompt:

```text
hooded knight holding a glowing sword, black armor, ancient ruins in the background
```

## 3. AI techniques used

### Stable Diffusion

Stable Diffusion is used as the main image generation model. It generates an image by starting from noise and gradually denoising it according to the text prompt.

### ControlNet

ControlNet is used to guide the generation process with an additional image input. In this project, the input sketch can be converted into an approximate OpenPose-style skeleton, a scribble line map, or a Canny edge map. Pose mode is used by default because it better matches the goal of preserving stick-figure body poses.

### Pose, Scribble, and Canny preprocessing

The input sketch is processed into a conditioning image. Pose mode converts strong stick-figure strokes into a colored skeleton on a black background. Scribble mode keeps rough black-on-white sketches as white guide lines. Canny preprocessing remains available for cleaner edge sketches.

### Prompt engineering

The project adds a dark fantasy style suffix to the user prompt. This makes the outputs more consistent with the intended visual direction.

### Optional LoRA adaptation (future goals)

A LoRA adapter can be used as an extension to specialize the model toward a dark fantasy character-art style. LoRA fine-tuning is more realistic than training a full diffusion model from scratch.

## 4. System pipeline

```text
User sketch + prompt
        ↓
Resize image to 1024x1024
        ↓
Apply Pose, Scribble, or Canny preprocessing
        ↓
Use the processed control image for conditioning
        ↓
Generate final image with Stable Diffusion
        ↓
Display output in Gradio
```

## 5. Implementation

The project is implemented in Python.

Main files:

| File | Purpose |
|---|---|
| `generate.py` | Core generation pipeline |
| `app.py` | Gradio user interface |
| `src/config.py` | Model IDs, prompts, presets |
| `requirements.txt` | Python dependencies |
| `docs/evaluation.md` | Evaluation of outputs |
| `docs/genai_usage.md` | Overview of GenAI usage |

The user can either run the command-line script or use the Gradio interface.

## 6. Evaluation plan

The system is evaluated using several sketches and prompts. Each result is judged on:

- Image quality
- How well the output follows the sketch
- How well the output follows the text prompt
- Dark fantasy style consistency
- Main failure points

## 7. Limitations

The system works best with clear full-body sketches. Very rough stick figures can still produce inconsistent anatomy. Extra clothing outlines and props can be interpreted as pose strokes in pose mode, so clean stick bodies usually preserve the pose best. Hands, weapons, and faces are sometimes distorted. The model also sometimes prioritizes the text prompt over the sketch, especially when the conditioning scale is too low.

## 8. Future work

Future improvements include:

- Training a custom LoRA on a curated dark fantasy character dataset
- Adding a web drawing canvas
- Improving sketch preprocessing
- Adding pose estimation
- Comparing outputs with and without LoRA
- Adding automatic prompt suggestions

## 9. Conclusion

The project demonstrates how text and visual conditioning can be combined to give users more control over AI image generation. Stable Diffusion provides the image generation capability, while ControlNet allows the user's sketch to influence the structure of the final image.
