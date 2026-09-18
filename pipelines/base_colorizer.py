"""
Base Colorization Pipeline using Stable Diffusion img2img
This provides the foundation for all other colorization tasks.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict
from diffusers import StableDiffusionImg2ImgPipeline
import warnings
warnings.filterwarnings("ignore")


class BaseColorizer:
    """
    Base colorization system using Stable Diffusion img2img pipeline.
    Converts grayscale images to color using generative AI.
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        """
        Initialize the base colorization pipeline.
        
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        print(f"🎨 Initializing Base Colorizer on {self.device}")
        print(f"📊 Using precision: {self.dtype}")
        
        self.pipe = self._load_pipeline(model_id)
        print("✅ Base Colorizer Ready!")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 GPU Detected: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("💻 Using CPU (GPU not available)")
        return torch.device(device)
    
    def _load_pipeline(self, model_id: str) -> StableDiffusionImg2ImgPipeline:
        """Load the Stable Diffusion img2img pipeline."""
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Memory optimizations
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("  ✓ XFormers Attention: Enabled")
            except Exception as e:
                print(f"  ⚠ XFormers: Not available ({e})")
            
            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                    print("  ✓ Full GPU Loading: Success")
                except RuntimeError as e:
                    print("  ⚠ GPU Memory Limited: Using CPU Offload")
                    pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()
                print("  ✓ CPU Sequential Offload: Enabled")
            
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load colorization pipeline: {e}")
    
    def colorize(
        self,
        image: Image.Image,
        prompt: str = "colorize this black and white photo, realistic colors, high quality",
        negative_prompt: str = "grayscale, black and white, monochrome, low quality, blurry",
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        Colorize a grayscale image.
        
        Args:
            image: Input PIL Image (can be grayscale or color)
            prompt: Text prompt to guide colorization
            negative_prompt: What to avoid in the output
            strength: How much to transform the image (0.0-1.0)
            guidance_scale: How closely to follow the prompt (1.0-20.0)
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (colorized_image, metadata_dict)
        """
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"🎨 Colorizing with prompt: '{prompt[:50]}...'")
        print(f"📏 Strength: {strength}, Steps: {num_inference_steps}, Seed: {seed}")
        
        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                )
            
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "steps": num_inference_steps,
                "seed": seed,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "input_size": image.size,
                "output_size": result.images[0].size
            }
            
            print("✅ Colorization complete!")
            return result.images[0], metadata
            
        except Exception as e:
            raise RuntimeError(f"Colorization failed: {str(e)}")
    
    def batch_colorize(
        self,
        images: list,
        prompt: str = "colorize this black and white photo, realistic colors",
        **kwargs
    ) -> list:
        """
        Colorize multiple images with the same settings.
        
        Args:
            images: List of PIL Images
            prompt: Colorization prompt
            **kwargs: Additional arguments for colorize()
        
        Returns:
            List of (colorized_image, metadata) tuples
        """
        results = []
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            colorized, metadata = self.colorize(img, prompt, **kwargs)
            results.append((colorized, metadata))
        return results


if __name__ == "__main__":
    # Test the base colorizer
    print("Testing Base Colorizer...")
    
    # Create a simple test image
    test_img = Image.new('RGB', (512, 512), color='gray')
    
    try:
        colorizer = BaseColorizer()
        colorized, metadata = colorizer.colorize(test_img)
        print("✅ Base colorizer test passed!")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"❌ Test failed: {e}")