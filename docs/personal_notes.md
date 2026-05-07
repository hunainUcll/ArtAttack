# my notes

## One-minute project explanation

Sketch2DarkFantasy is a prototype that generates dark fantasy character concept art from a rough sketch and a text prompt. The sketch controls the pose and composition, while the prompt controls the character details and style. The system uses Stable Diffusion for generation and ControlNet to condition the generation on the sketch.

## Why this project is useful

Many users have a creative idea for a character but cannot draw it. Text-to-image tools are powerful, but they do not give precise control over pose or composition. Our system gives the user more control by combining sketch input with text input.

## Why ControlNet?

Stable Diffusion normally generates images mainly from text. ControlNet adds an extra conditioning input, such as an edge map. This allows the model to follow the structure of the input sketch more closely.

## What does Canny do?

Canny edge detection converts the sketch into a clearer edge map. This edge map is used as the ControlNet conditioning image.

## What does the prompt do?

The prompt defines semantic and stylistic details, such as the character class, weapon, armor, lighting, and environment.

## What is LoRA?

LoRA is a lightweight fine-tuning method. Instead of training the entire Stable Diffusion model, it trains a small adapter that can influence the model's style or subject.

## Did we train Stable Diffusion from scratch?

No. Training Stable Diffusion from scratch would require a huge dataset and very large compute resources. This project uses pretrained Stable Diffusion and ControlNet, and includes LoRA adaptation as a realistic extension for specializing the style.

## Main difficulties

- Getting useful results from very rough sketches.
- Avoiding distorted anatomy.
- Making weapons and hands look correct.
- Balancing prompt strength and sketch strength.

## What was learned

- How a diffusion pipeline is loaded with Hugging Face Diffusers.
- How ControlNet uses a conditioning image.
- How preprocessing affects generated results.
- How parameters such as seed, guidance scale, and ControlNet scale affect output.
- Why fine-tuning with LoRA is more realistic than training a full model.
