"""
Conditional Image Colorization Pipeline for Task 2
Allows users to specify desired colors for specific objects/regions.

Author: Sumaiya Ibrahim
Task: Task 2 - Conditional Image Colorization
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
import cv2
import re

from pipelines.base_colorizer import BaseColorizer
from segmentation.semantic_segmenter import SemanticSegmenter
from processing.image_processor import ImageProcessor


class ConditionalColorizer:
    """
    Conditional colorization system that allows users to specify colors for specific objects.
    Integrates semantic segmentation with user-defined color conditions.
    """
    
    # Predefined color mappings for common objects
    COLOR_PRESETS = {
        'sky': {
            'blue': (135, 206, 235),
            'light_blue': (173, 216, 230),
            'dark_blue': (25, 25, 112),
            'sunset': (255, 127, 80),
            'night': (25, 25, 112)
        },
        'grass': {
            'green': (34, 139, 34),
            'light_green': (144, 238, 144),
            'yellow_green': (154, 205, 50),
            'dry': (201, 180, 140)
        },
        'car': {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'silver': (192, 192, 192)
        },
        'building': {
            'brick': (178, 34, 34),
            'concrete': (128, 128, 128),
            'stone': (105, 105, 105),
            'wood': (139, 69, 19)
        },
        'tree': {
            'green': (34, 139, 34),
            'autumn': (255, 140, 0),
            'pine': (0, 100, 0),
            'palm': (107, 142, 35)
        },
        'water': {
            'blue': (0, 105, 148),
            'clear': (64, 164, 223),
            'dark': (0, 0, 139),
            'turquoise': (64, 224, 208)
        },
        'clothing': {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 128, 0),
            'yellow': (255, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        },
        'skin': {
            'light': (255, 224, 189),
            'medium': (255, 200, 150),
            'dark': (180, 130, 90),
            'olive': (200, 160, 120)
        }
    }
    
    def __init__(self, device: str = "auto"):
        """
        Initialize conditional colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"🎨 Initializing Conditional Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize semantic segmenter for object detection
            self.segmenter = SemanticSegmenter(device=device)
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            print("✅ Conditional Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize conditional colorizer: {e}")
            raise
    
    def parse_color_conditions(self, conditions_text: str) -> Dict[str, Tuple[int, int, int]]:
        """
        Parse user-provided color conditions from text.
        
        Examples:
        - "sky:blue, grass:green, car:red"
        - "Make the sky blue and the grass green"
        - "sky=blue, grass=green, car=red"
        
        Args:
            conditions_text: User input describing color conditions
        
        Returns:
            Dictionary mapping object names to RGB colors
        """
        conditions = {}
        
        if not conditions_text or not conditions_text.strip():
            return conditions
        
        # Try different parsing patterns
        patterns = [
            r'(\w+):\s*(\w+)',           # "sky: blue"
            r'(\w+)\s*=\s*(\w+)',        # "sky = blue"
            r'(\w+)\s+(?:to|as|is)\s+(\w+)',  # "sky to blue", "sky as blue"
            r'make\s+(\w+)\s+(\w+)',     # "make sky blue"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, conditions_text.lower())
            for obj, color in matches:
                if obj in self.COLOR_PRESETS and color in self.COLOR_PRESETS[obj]:
                    conditions[obj] = self.COLOR_PRESETS[obj][color]
                elif obj in self.COLOR_PRESETS:
                    # Try to find the closest color match
                    conditions[obj] = self._find_closest_color(obj, color)
        
        return conditions
    
    def _find_closest_color(self, object_type: str, color_name: str) -> Tuple[int, int, int]:
        """
        Find the closest matching color for an object type.
        
        Args:
            object_type: Type of object (sky, grass, etc.)
            color_name: Name of the desired color
        
        Returns:
            RGB tuple for the closest matching color
        """
        if object_type not in self.COLOR_PRESETS:
            # Return a default color
            return (128, 128, 128)
        
        colors = self.COLOR_PRESETS[object_type]
        
        # Simple color name matching
        color_name = color_name.lower()
        for color_key, color_value in colors.items():
            if color_name in color_key or color_key in color_name:
                return color_value
        
        # Return first available color as default
        return list(colors.values())[0]
    
    def apply_color_conditions(
        self,
        image: Image.Image,
        conditions: Dict[str, Tuple[int, int, int]],
        use_segmentation: bool = True
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply user-defined color conditions to the image.
        
        Args:
            image: Input PIL Image
            conditions: Dictionary of object->color mappings
            use_segmentation: Whether to use semantic segmentation
        
        Returns:
            Tuple of (colorized_image, metadata_dict)
        """
        metadata = {
            'conditions_applied': conditions,
            'method': 'conditional_colorization'
        }
        
        if not conditions:
            # No conditions, return basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['colorization_metadata'] = color_metadata
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
        
        try:
            if use_segmentation:
                # Use semantic segmentation to identify regions
                mask, detected_objects = self.segmenter.segment_image(image)
                metadata['segmentation_results'] = detected_objects
                
                # Create modified image with applied conditions
                img_array = np.array(image).copy()
                
                # Map user conditions to detected objects
                for obj_name, desired_color in conditions.items():
                    # Find matching objects in segmentation
                    matching_classes = self._find_matching_objects(obj_name, detected_objects)
                    
                    for class_info in matching_classes:
                        class_id = class_info['class_id']
                        # Apply color to matching regions
                        region_mask = (mask == class_id)
                        if np.any(region_mask):
                            img_array[region_mask] = desired_color
                
                # Apply base colorization to preserve overall quality
                modified_image = Image.fromarray(img_array)
                colorized, color_metadata = self.colorizer.colorize(modified_image)
                
                # Blend the conditional colors with the colorized result
                final_image = self._blend_colorized_with_conditions(
                    colorized, img_array, mask, conditions, detected_objects
                )
                
                metadata['colorization_metadata'] = color_metadata
                return final_image, metadata
                
            else:
                # Simple colorization with modified prompt
                prompt = self._create_conditional_prompt(conditions)
                colorized, color_metadata = self.colorizer.colorize(
                    image, 
                    prompt=prompt
                )
                metadata['colorization_metadata'] = color_metadata
                metadata['prompt_used'] = prompt
                return colorized, metadata
                
        except Exception as e:
            print(f"⚠️ Conditional colorization error: {e}")
            metadata['error'] = str(e)
            # Fallback to basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
    
    def _find_matching_objects(
        self, 
        condition_object: str, 
        detected_objects: Dict
    ) -> List[Dict]:
        """
        Find objects in segmentation results that match the user's condition.
        
        Args:
            condition_object: Object name from user condition
            detected_objects: Segmentation results
        
        Returns:
            List of matching object information
        """
        matching = []
        condition_lower = condition_object.lower()
        
        for obj in detected_objects['objects']:
            class_name = obj['class_name'].lower()
            category = obj['category'].lower()
            
            # Check for name match
            if condition_object in class_name or class_name in condition_lower:
                matching.append(obj)
            # Check for category match
            elif condition_object in category or category in condition_lower:
                matching.append(obj)
        
        return matching
    
    def _create_conditional_prompt(self, conditions: Dict[str, Tuple[int, int, int]]) -> str:
        """
        Create a colorization prompt based on user conditions.
        
        Args:
            conditions: Dictionary of object->color mappings
        
        Returns:
            Enhanced prompt for colorization
        """
        base_prompt = "colorize this black and white photo, realistic colors, high quality"
        
        condition_prompts = []
        for obj, color in conditions.items():
            color_name = self._get_color_name(color)
            condition_prompts.append(f"{obj} should be {color_name}")
        
        if condition_prompts:
            enhanced_prompt = base_prompt + ", " + ", ".join(condition_prompts)
            return enhanced_prompt
        
        return base_prompt
    
    def _get_color_name(self, color: Tuple[int, int, int]) -> str:
        """
        Get a descriptive name for an RGB color.
        
        Args:
            color: RGB tuple
        
        Returns:
            Color name string
        """
        # Simple color name mapping
        color_names = {
            (255, 0, 0): 'red',
            (0, 255, 0): 'green',
            (0, 0, 255): 'blue',
            (255, 255, 0): 'yellow',
            (255, 0, 255): 'magenta',
            (0, 255, 255): 'cyan',
            (255, 255, 255): 'white',
            (0, 0, 0): 'black',
            (128, 128, 128): 'gray',
            (135, 206, 235): 'sky blue',
            (34, 139, 34): 'forest green'
        }
        
        # Find closest match
        for rgb, name in color_names.items():
            if all(abs(c - t) < 30 for c, t in zip(color, rgb)):
                return name
        
        return f"RGB{color}"
    
    def _blend_colorized_with_conditions(
        self,
        colorized: Image.Image,
        condition_image: np.ndarray,
        mask: np.ndarray,
        conditions: Dict[str, Tuple[int, int, int]],
        detected_objects: Dict
    ) -> Image.Image:
        """
        Blend the AI colorized result with user-specified conditional colors.
        
        Args:
            colorized: AI colorized image
            condition_image: Image with user colors applied
            mask: Segmentation mask
            conditions: User color conditions
            detected_objects: Segmentation results
        
        Returns:
            Blended final image
        """
        colorized_array = np.array(colorized)
        blended = colorized_array.copy()
        
        # For each condition, blend the user color with AI colorization
        for obj_name, desired_color in conditions.items():
            matching_classes = self._find_matching_objects(obj_name, detected_objects)
            
            for class_info in matching_classes:
                class_id = class_info['class_id']
                region_mask = (mask == class_id)
                
                if np.any(region_mask):
                    # Blend user color with AI colorization (70% user, 30% AI)
                    user_color = np.array(desired_color)
                    blended[region_mask] = (
                        0.7 * condition_image[region_mask] + 
                        0.3 * colorized_array[region_mask]
                    ).astype(np.uint8)
        
        return Image.fromarray(blended)
    
    def create_colorization_preview(
        self,
        original: Image.Image,
        conditions: Dict[str, Tuple[int, int, int]]
    ) -> Image.Image:
        """
        Create a preview showing where colors will be applied.
        
        Args:
            original: Original image
            conditions: User color conditions
        
        Returns:
            Preview image with color indicators
        """
        try:
            mask, detected_objects = self.segmenter.segment_image(original)
            preview = original.copy()
            draw = ImageDraw.Draw(preview)
            
            # For each condition, highlight matching regions
            for obj_name, desired_color in conditions.items():
                matching_classes = self._find_matching_objects(obj_name, detected_objects)
                
                for class_info in matching_classes:
                    class_id = class_info['class_id']
                    region_mask = (mask == class_id)
                    
                    if np.any(region_mask):
                        # Create a colored overlay for the region
                        overlay = Image.new('RGBA', original.size, desired_color + (128,))
                        mask_image = Image.fromarray((region_mask * 255).astype(np.uint8))
                        overlay.putalpha(mask_image)
                        
                        # Blend with preview
                        preview = Image.alpha_composite(
                            preview.convert('RGBA'),
                            overlay
                        ).convert('RGB')
            
            return preview
            
        except Exception as e:
            print(f"⚠️ Preview creation error: {e}")
            return original
    
    def get_available_objects(self) -> List[str]:
        """
        Get list of available objects for conditional colorization.
        
        Returns:
            List of object names that can be colorized
        """
        return list(self.COLOR_PRESETS.keys())
    
    def get_available_colors(self, object_type: str) -> List[str]:
        """
        Get available colors for a specific object type.
        
        Args:
            object_type: Type of object (sky, grass, etc.)
        
        Returns:
            List of available color names
        """
        if object_type in self.COLOR_PRESETS:
            return list(self.COLOR_PRESETS[object_type].keys())
        return []


if __name__ == "__main__":
    # Test conditional colorizer
    print("Testing Conditional Colorizer...")
    
    try:
        colorizer = ConditionalColorizer()
        
        # Test color condition parsing
        test_conditions = colorizer.parse_color_conditions("sky:blue, grass:green, car:red")
        print(f"✅ Parsed conditions: {test_conditions}")
        
        # Test with image
        test_img = Image.new('RGB', (512, 512), color='gray')
        
        # Test conditional colorization
        colorized, metadata = colorizer.apply_color_conditions(test_img, test_conditions)
        print(f"✅ Conditional colorization complete! Metadata keys: {metadata.keys()}")
        
        # Test available objects and colors
        objects = colorizer.get_available_objects()
        print(f"✅ Available objects: {objects}")
        
        sky_colors = colorizer.get_available_colors('sky')
        print(f"✅ Available sky colors: {sky_colors}")
        
        print("✅ Conditional colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")