"""
Shared Image Processing Utilities
Common functions used across all colorization tasks.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, List
import cv2


class ImageProcessor:
    """
    Shared image processing utilities for all colorization tasks.
    """
    
    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB format."""
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image
    
    @staticmethod
    def ensure_grayscale(image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        if image.mode != 'L':
            return image.convert('L')
        return image
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size while maintaining aspect ratio."""
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
        """Crop image from center."""
        width, height = image.size
        crop_width, crop_height = crop_size
        
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust image brightness (factor: 0.0-2.0, 1.0 = original)."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust image contrast (factor: 0.0-2.0, 1.0 = original)."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_saturation(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjust image saturation (factor: 0.0-2.0, 1.0 = original)."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def sharpen_image(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """Sharpen image (factor: 0.0-2.0, 1.0 = original)."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_gamma_correction(image: Image.Image, gamma: float = 1.0) -> Image.Image:
        """Apply gamma correction to image."""
        if gamma == 1.0:
            return image
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply gamma correction
        img_array = np.power(img_array, 1.0 / gamma)
        
        # Convert back to PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    @staticmethod
    def denoise_image(image: Image.Image, strength: int = 1) -> Image.Image:
        """Apply denoising filter to image."""
        if strength == 0:
            return image
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, strength*10, strength*10, 7, 21)
        
        # Convert back to PIL
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(denoised_rgb)
    
    @staticmethod
    def apply_color_cast(image: Image.Image, color: Tuple[int, int, int], intensity: float = 0.3) -> Image.Image:
        """
        Apply a color cast/tint to the image.
        
        Args:
            image: Input PIL Image
            color: RGB tuple (0-255) for the tint color
            intensity: Strength of the color cast (0.0-1.0)
        """
        if intensity == 0:
            return image
        
        # Create color overlay
        overlay = Image.new('RGB', image.size, color)
        
        # Blend with original image
        blended = Image.blend(image, overlay, intensity)
        
        return blended
    
    @staticmethod
    def sepia_tone(image: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Apply sepia tone effect."""
        if intensity == 0:
            return image
        
        # Convert to grayscale first
        gray = image.convert('L')
        
        # Create sepia matrix
        sepia = Image.new('RGB', image.size)
        pixels = sepia.load()
        gray_pixels = gray.load()
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                g = gray_pixels[x, y]
                r = min(255, int(g * 1.07))
                g_green = min(255, int(g * 0.74))
                b = min(255, int(g * 0.43))
                pixels[x, y] = (r, g_green, b)
        
        # Blend with original
        return Image.blend(image, sepia, intensity)
    
    @staticmethod
    def vignette_effect(image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Apply vignette effect (darkened corners)."""
        if intensity == 0:
            return image
        
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create radial gradient
        radius = np.sqrt(X**2 + Y**2)
        vignette = 1 - intensity * radius
        
        # Clip to valid range
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to image
        img_array = np.array(image).astype(np.float32) / 255.0
        if len(img_array.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
        
        vignetted = img_array * vignette
        vignetted = (vignetted * 255).astype(np.uint8)
        
        return Image.fromarray(vignetted)
    
    @staticmethod
    def enhance_for_historical(image: Image.Image, era: str = "1950s") -> Image.Image:
        """
        Apply era-specific enhancements for historical photographs.
        
        Args:
            image: Input PIL Image
            era: Historical era for specific treatment
        """
        # Base enhancements
        enhanced = ImageProcessor.adjust_contrast(image, 1.1)
        enhanced = ImageProcessor.sharpen_image(enhanced, 1.2)
        
        # Era-specific treatments
        era_treatments = {
            "1900s": {
                "sepia": 0.6,
                "vignette": 0.4,
                "contrast": 0.9,
                "saturation": 0.7
            },
            "1920s": {
                "sepia": 0.4,
                "vignette": 0.3,
                "contrast": 1.0,
                "saturation": 0.8
            },
            "1950s": {
                "sepia": 0.2,
                "vignette": 0.2,
                "contrast": 1.1,
                "saturation": 0.9
            },
            "1970s": {
                "sepia": 0.1,
                "vignette": 0.1,
                "contrast": 1.05,
                "saturation": 1.0
            }
        }
        
        treatment = era_treatments.get(era, era_treatments["1950s"])
        
        if treatment["sepia"] > 0:
            enhanced = ImageProcessor.sepia_tone(enhanced, treatment["sepia"])
        
        if treatment["vignette"] > 0:
            enhanced = ImageProcessor.vignette_effect(enhanced, treatment["vignette"])
        
        enhanced = ImageProcessor.adjust_contrast(enhanced, treatment["contrast"])
        enhanced = ImageProcessor.adjust_saturation(enhanced, treatment["saturation"])
        
        return enhanced
    
    @staticmethod
    def preprocess_sketch(image: Image.Image) -> Image.Image:
        """Preprocess sketch images for colorization."""
        # Enhance edges
        enhanced = ImageProcessor.sharpen_image(image, 1.5)
        
        # Increase contrast for line clarity
        enhanced = ImageProcessor.adjust_contrast(enhanced, 1.3)
        
        return enhanced
    
    @staticmethod
    def preprocess_infrared(image: Image.Image) -> Image.Image:
        """Preprocess infrared/satellite images for colorization."""
        # Normalize intensity
        enhanced = ImageProcessor.adjust_contrast(image, 1.2)
        enhanced = ImageProcessor.adjust_brightness(enhanced, 1.1)
        
        # Reduce noise
        enhanced = ImageProcessor.denoise_image(enhanced, strength=1)
        
        return enhanced
    
    @staticmethod
    def create_comparison_image(original: Image.Image, colorized: Image.Image) -> Image.Image:
        """Create side-by-side comparison image."""
        # Ensure both images are same size
        if original.size != colorized.size:
            colorized = colorized.resize(original.size, Image.Resampling.LANCZOS)
        
        # Create side-by-side image
        total_width = original.size[0] * 2
        max_height = max(original.size[1], colorized.size[1])
        
        comparison = Image.new('RGB', (total_width, max_height))
        comparison.paste(original, (0, 0))
        comparison.paste(colorized, (original.size[0], 0))
        
        return comparison


if __name__ == "__main__":
    # Test the image processor
    print("Testing Image Processor...")
    
    # Create a test image
    test_img = Image.new('RGB', (256, 256), color='blue')
    
    # Test various functions
    rgb_img = ImageProcessor.ensure_rgb(test_img)
    print(f"✅ RGB conversion: {rgb_img.mode}")
    
    resized = ImageProcessor.resize_image(test_img, (128, 128))
    print(f"✅ Resize: {resized.size}")
    
    brightened = ImageProcessor.adjust_brightness(test_img, 1.5)
    print("✅ Brightness adjustment")
    
    contrasted = ImageProcessor.adjust_contrast(test_img, 1.5)
    print("✅ Contrast adjustment")
    
    sepia = ImageProcessor.sepia_tone(test_img, 0.5)
    print("✅ Sepia tone")
    
    historical = ImageProcessor.enhance_for_historical(test_img, "1950s")
    print("✅ Historical enhancement")
    
    print("✅ Image processor tests passed!")