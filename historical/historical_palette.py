"""
Historical Color Palette Module for Task 4
Provides era-specific color palettes and color treatments for historical photographs.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import colorsys


class HistoricalPalette:
    """
    Historical color palettes for different eras.
    Provides era-specific color treatments and transformations.
    """
    
    # Era-specific color palettes
    ERA_PALETTES = {
        '1900s': {
            'name': 'Early 1900s Victorian',
            'primary_colors': [
                (139, 69, 19),    # Sepia brown
                (160, 82, 45),    # Sienna
                (210, 180, 140),  # Tan
                (128, 128, 128),  # Gray
                (245, 245, 220)   # Beige
            ],
            'secondary_colors': [
                (112, 66, 20),    # Dark brown
                (188, 143, 143),  # Rosy brown
                (169, 169, 169),  # Dark gray
                (222, 184, 135),  # Burlywood
                (244, 164, 96)    # Sandy brown
            ],
            'tone_characteristics': {
                'warmth': 0.8,      # High warmth
                'saturation': 0.3,  # Low saturation
                'contrast': 0.9,   # Slightly low contrast
                'brightness': 0.85  # Slightly dim
            }
        },
        '1920s': {
            'name': '1920s Roaring Twenties',
            'primary_colors': [
                (205, 133, 63),   # Peru
                (210, 105, 30),   # Chocolate
                (188, 143, 143),  # Rosy brown
                (119, 136, 153),  # Light slate gray
                (255, 228, 196)   # Moccasin
            ],
            'secondary_colors': [
                (160, 82, 45),    # Sienna
                (222, 184, 135),  # Burlywood
                (176, 196, 222),  # Light steel blue
                (238, 232, 170),  # Pale goldenrod
                (255, 182, 193)   # Light pink
            ],
            'tone_characteristics': {
                'warmth': 0.7,      # Medium-high warmth
                'saturation': 0.4,  # Low-medium saturation
                'contrast': 1.0,   # Normal contrast
                'brightness': 0.9   # Normal brightness
            }
        },
        '1950s': {
            'name': '1950s Post-War',
            'primary_colors': [
                (255, 140, 0),    # Dark orange
                (34, 139, 34),    # Forest green
                (70, 130, 180),   # Steel blue
                (255, 215, 0),    # Gold
                (220, 20, 60)     # Crimson
            ],
            'secondary_colors': [
                (255, 99, 71),    # Tomato
                (60, 179, 113),   # Medium sea green
                (100, 149, 237),  # Cornflower blue
                (255, 165, 0),    # Orange
                (147, 112, 219)   # Medium purple
            ],
            'tone_characteristics': {
                'warmth': 0.5,      # Medium warmth
                'saturation': 0.6,  # Medium saturation
                'contrast': 1.1,   # Slightly high contrast
                'brightness': 1.0   # Normal brightness
            }
        },
        '1970s': {
            'name': '1970s Vintage Film',
            'primary_colors': [
                (255, 127, 80),   # Coral
                (218, 165, 32),   # Golden rod
                (154, 205, 50),   # Yellow green
                (255, 105, 180),  # Hot pink
                (0, 206, 209)     # Dark turquoise
            ],
            'secondary_colors': [
                (255, 160, 122),  # Light salmon
                (240, 230, 140),  # Khaki
                (32, 178, 170),   # Light sea green
                (255, 182, 193),  # Light pink
                (175, 238, 238)   # Pale turquoise
            ],
            'tone_characteristics': {
                'warmth': 0.6,      # Medium-high warmth
                'saturation': 0.7,  # Medium-high saturation
                'contrast': 1.0,   # Normal contrast
                'brightness': 1.05  # Slightly bright
            }
        }
    }
    
    def __init__(self):
        """Initialize the historical palette system."""
        print("🎨 Initializing Historical Color Palette System")
        print("✅ Historical Palette System Ready!")
    
    def get_era_palette(self, era: str) -> Optional[Dict]:
        """
        Get the color palette for a specific era.
        
        Args:
            era: Era string (e.g., '1920s', '1950s')
        
        Returns:
            Palette dictionary or None if era not found
        """
        return self.ERA_PALETTES.get(era)
    
    def apply_era_color_treatment(
        self,
        image: Image.Image,
        era: str,
        intensity: float = 0.7
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply era-specific color treatment to an image.
        
        Args:
            image: Input PIL Image
            era: Historical era
            intensity: Treatment intensity (0.0-1.0)
        
        Returns:
            Tuple of (treated_image, metadata_dict)
        """
        metadata = {
            'era': era,
            'intensity': intensity,
            'method': 'era_color_treatment'
        }
        
        if era not in self.ERA_PALETTES:
            metadata['error'] = f'Era {era} not found'
            return image, metadata
        
        try:
            palette = self.ERA_PALETTES[era]
            tone_chars = palette['tone_characteristics']
            
            # Convert to numpy array for processing
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Apply era-specific tone adjustments
            img_array = self._apply_tone_adjustments(img_array, tone_chars, intensity)
            
            # Apply era-specific color cast
            img_array = self._apply_era_color_cast(img_array, palette, intensity)
            
            # Convert back to PIL Image
            treated_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
            treated_image = Image.fromarray(treated_array)
            
            metadata['palette_used'] = palette['name']
            metadata['tone_adjustments'] = tone_chars
            
            return treated_image, metadata
            
        except Exception as e:
            print(f"⚠️ Era color treatment error: {e}")
            metadata['error'] = str(e)
            return image, metadata
    
    def _apply_tone_adjustments(
        self,
        img_array: np.ndarray,
        tone_chars: Dict,
        intensity: float
    ) -> np.ndarray:
        """
        Apply tone adjustments based on era characteristics.
        
        Args:
            img_array: Image as float array (0.0-1.0)
            tone_chars: Tone characteristics dictionary
            intensity: Adjustment intensity
        
        Returns:
            Adjusted image array
        """
        # Warmth adjustment (red/blue balance)
        warmth = tone_chars.get('warmth', 0.5)
        if warmth > 0.5:
            # Add warmth (increase red, decrease blue)
            img_array[:, :, 0] *= (1 + (warmth - 0.5) * intensity * 0.3)  # Red
            img_array[:, :, 2] *= (1 - (warmth - 0.5) * intensity * 0.2)  # Blue
        elif warmth < 0.5:
            # Add coolness (decrease red, increase blue)
            img_array[:, :, 0] *= (1 - (0.5 - warmth) * intensity * 0.2)  # Red
            img_array[:, :, 2] *= (1 + (0.5 - warmth) * intensity * 0.3)  # Blue
        
        # Saturation adjustment
        saturation = tone_chars.get('saturation', 0.5)
        if saturation != 0.5:
            # Convert to HSV for saturation adjustment
            hsv = self._rgb_to_hsv(img_array)
            hsv[:, :, 1] *= (0.5 + saturation) * intensity
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
            img_array = self._hsv_to_rgb(hsv)
        
        # Contrast adjustment
        contrast = tone_chars.get('contrast', 1.0)
        if contrast != 1.0:
            # Apply contrast using sigmoid-like function
            img_array = (img_array - 0.5) * contrast + 0.5
        
        # Brightness adjustment
        brightness = tone_chars.get('brightness', 1.0)
        if brightness != 1.0:
            img_array *= brightness
        
        return img_array
    
    def _apply_era_color_cast(
        self,
        img_array: np.ndarray,
        palette: Dict,
        intensity: float
    ) -> np.ndarray:
        """
        Apply a subtle color cast based on era palette.
        
        Args:
            img_array: Image as float array
            palette: Era palette dictionary
            intensity: Cast intensity
        
        Returns:
            Image with color cast applied
        """
        # Get primary colors from palette
        primary_colors = palette.get('primary_colors', [])
        if not primary_colors:
            return img_array
        
        # Calculate average color from palette
        avg_color = np.mean([np.array(color) / 255.0 for color in primary_colors], axis=0)
        
        # Apply subtle color cast
        img_array = img_array * (1 - intensity * 0.1) + avg_color * intensity * 0.1
        
        return img_array
    
    def _rgb_to_hsv(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB array to HSV."""
        hsv_array = np.zeros_like(rgb_array)
        for i in range(rgb_array.shape[0]):
            for j in range(rgb_array.shape[1]):
                r, g, b = rgb_array[i, j]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_array[i, j] = [h, s, v]
        return hsv_array
    
    def _hsv_to_rgb(self, hsv_array: np.ndarray) -> np.ndarray:
        """Convert HSV array to RGB."""
        rgb_array = np.zeros_like(hsv_array)
        for i in range(hsv_array.shape[0]):
            for j in range(hsv_array.shape[1]):
                h, s, v = hsv_array[i, j]
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                rgb_array[i, j] = [r, g, b]
        return rgb_array
    
    def create_palette_preview(self, era: str, size: Tuple[int, int] = (400, 100)) -> Image.Image:
        """
        Create a preview image showing the era's color palette.
        
        Args:
            era: Era string
            size: Size of preview image
        
        Returns:
            Palette preview image
        """
        if era not in self.ERA_PALETTES:
            return Image.new('RGB', size, color='gray')
        
        palette = self.ERA_PALETTES[era]
        colors = palette['primary_colors']
        
        # Create palette preview
        preview = Image.new('RGB', size)
        draw_image = Image.new('RGBA', size)
        
        # Draw color swatches
        num_colors = len(colors)
        swatch_width = size[0] // num_colors
        
        for i, color in enumerate(colors):
            x_start = i * swatch_width
            x_end = (i + 1) * swatch_width
            swatch = Image.new('RGB', (swatch_width, size[1]), color)
            preview.paste(swatch, (x_start, 0))
        
        return preview
    
    def get_color_suggestions(self, era: str, category: str = 'primary') -> List[Tuple[int, int, int]]:
        """
        Get color suggestions for a specific era and category.
        
        Args:
            era: Era string
            category: 'primary' or 'secondary'
        
        Returns:
            List of RGB color tuples
        """
        if era not in self.ERA_PALETTES:
            return []
        
        palette = self.ERA_PALETTES[era]
        
        if category == 'primary':
            return palette.get('primary_colors', [])
        elif category == 'secondary':
            return palette.get('secondary_colors', [])
        
        return []


if __name__ == "__main__":
    # Test historical palette
    print("Testing Historical Palette...")
    
    try:
        palette = HistoricalPalette()
        
        # Test getting era palette
        era_1920s = palette.get_era_palette('1920s')
        print(f"✅ 1920s palette: {era_1920s['name']}")
        
        # Test applying era treatment
        test_img = Image.new('RGB', (512, 512), color='blue')
        treated, metadata = palette.apply_era_color_treatment(test_img, '1920s')
        print(f"✅ Era treatment applied! Metadata keys: {metadata.keys()}")
        
        # Test palette preview
        preview = palette.create_palette_preview('1950s')
        print(f"✅ Palette preview created: {preview.size}")
        
        # Test color suggestions
        colors = palette.get_color_suggestions('1970s', 'primary')
        print(f"✅ 1970s primary colors: {len(colors)} colors")
        
        print("✅ Historical palette tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")