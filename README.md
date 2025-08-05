# Stable Diffusion Image Generation Pipeline
A custom implementation of Stable Diffusion for text-to-image and image-to-image generation using PyTorch and Hugging Face's diffusers library, optimized for Apple Silicon Macs.

## Overview
This notebook implements a complete diffusion pipeline from scratch, providing full control over the image generation process with real-time monitoring and debugging capabilities.

## Features
* Text-to-Image Generation: Create images from text prompts using CLIP embeddings
* Image-to-Image Generation: Transform existing images based on text descriptions
* Apple Silicon Optimization: Leverages MPS (Metal Performance Shaders) for GPU acceleration
* Real-time Monitoring: Track latent norm progression and detect generation instability
* Memory Management: Efficient memory usage with automatic cache clearing
* Intermediate Visualization: Save and display images at each denoising step
* Classifier-free Guidance: Enhanced prompt adherence and image quality

## Core Components

* CLIP: Text Encoder; Converts text prompts to embeddings
* VAE: (Variational Autoencoder); Encodes/decodes between image and latent space
* U-Net: Core denoising model for diffusion process
* LMS Scheduler: Controls noise removal schedule

## Installation

```pip install torch torchvision transformers diffusers pillow numpy tqdm python-dotenv```

## Usage
### Text-to-Image Generation

```
images = generate_image(
    prompts=["A sunset over mountains"], 
    g=7.5,          # Guidance scale
    seed=42,        # Random seed
    steps=50,       # Denoising steps
    save_int=True   # Save intermediate images
)
```

### Image-to-Image Generation

```
images = generate_image(
    prompts=["Turn this into a Van Gogh painting"],
    input_image="photo.jpg",
    strength=0.7,   # Transformation strength (0.0-1.0)
    g=7.5,
```

## Parameters
### Parameter / Range / Description
* g:	1.0-20.0;	Guidance scale (higher = more prompt adherence)
* strength:	0.0-1.0;	Image transformation strength (img2img only)
* steps:	10-100;	Number of denoising steps (more = higher quality)
* seed:	Any int;	Random seed for reproducible results
* input_image: path to image or None; If None, then director text-to-image will be employed

## Key Functions
* generate_image(): Main generation function with full parameter control
* text_encode(): Convert prompts to CLIP embeddings
* convert_pil_to_latents(): Encode images to latent space
* convert_latents_to_pil(): Decode latents back to viewable images
* clear_memory(): Memory management for Apple Silicon

## Monitoring & Debugging
The pipeline includes built-in monitoring:

* Latent Norm Tracking: Detect generation instability (healthy range: 3-6)
* Divergence Detection: Automatic warnings for unstable generation
* Progress Visualization: Real-time image evolution display
* Memory Usage: Automatic GPU memory management

## Example Output
TBD

## Hardware Requirements
Apple Silicon Mac (M1/M2/M3)
macOS with Metal support
Minimum 8GB unified memory (16GB+ recommended)

## Model Details
Base Model: Stable Diffusion v1.4
Precision: Float16 for memory efficiency
Device: MPS (Metal Performance Shaders)
Resolution: 512x512 (configurable)

## License
This implementation uses pre-trained models from Hugging Face. Please refer to individual model licenses for usage terms.
