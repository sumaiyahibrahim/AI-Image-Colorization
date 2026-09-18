"""
Context-Aware Colorization Pipeline for Task 3
Handles complex scenes with contextual relationships between objects.

Author: Sumaiya Ibrahim
Task: Task 3 - Context-Aware Colorization of Complex Scenes
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2

from pipelines.base_colorizer import BaseColorizer
from segmentation.semantic_segmenter import SemanticSegmenter
from processing.image_processor import ImageProcessor


class ContextColorizer:
    """
    Context-aware colorization for complex scenes.
    Considers relationships between objects (sky/background, roads, buildings, trees, shadows, reflections).
    """
    
    # Scene context rules for color relationships
    SCENE_CONTEXT_RULES = {
        'outdoor': {
            'sky_to_ground_transition': True,
            'lighting_consistency': True,
            'shadow_consistency': True,
            'natural_color_harmony': True
        },
        'urban': {
            'building_color_consistency': True,
            'road_surface_consistency': True,
            'vehicle_color_realism': True,
            'artificial_lighting': True
        },
        'indoor': {
            'wall_color_consistency': True,
            'furniture_color_harmony': True,
            'lighting_temperature': True,
            'shadow_direction': True
        },
        'nature': {
            'vegetation_color_consistency': True,
            'water_color_realism': True,
            'natural_lighting': True,
            'seasonal_consistency': True
        }
    }
    
    # Color harmony palettes for different scene types
    COLOR_HARMONY_PALETTES = {
        'outdoor': {
            'sky': [(135, 206, 235), (176, 224, 230), (25, 25, 112)],
            'vegetation': [(34, 139, 34), (107, 142, 35), (85, 107, 47)],
            'ground': [(139, 69, 19), (160, 82, 45), (210, 180, 140)],
            'water': [(0, 105, 148), (64, 164, 223), (0, 0, 139)]
        },
        'urban': {
            'buildings': [(128, 128, 128), (169, 169, 169), (105, 105, 105)],
            'roads': [(105, 105, 105), (128, 128, 128), (169, 169, 169)],
            'vehicles': [(255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)],
            'signs': [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
        },
        'nature': {
            'trees': [(34, 139, 34), (0, 100, 0), (107, 142, 35)],
            'water': [(0, 105, 148), (64, 164, 223), (32, 178, 170)],
            'rocks': [(128, 128, 128), (105, 105, 105), (169, 169, 169)],
            'sky': [(135, 206, 235), (176, 224, 230), (255, 127, 80)]
        }
    }
    
    def __init__(self, device: str = "auto"):
        """
        Initialize context-aware colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"🏙️ Initializing Context-Aware Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize semantic segmenter for scene understanding
            self.segmenter = SemanticSegmenter(device=device)
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            print("✅ Context-Aware Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize context colorizer: {e}")
            raise
    
    def analyze_scene_context(self, image: Image.Image) -> Dict:
        """
        Analyze the scene to determine context type and object relationships.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Dictionary with scene context information
        """
        context = {
            'scene_type': 'outdoor',  # Default
            'dominant_objects': [],
            'object_relationships': [],
            'lighting_estimate': 'neutral',
            'color_harmony_suggestions': {}
        }
        
        try:
            # Perform semantic segmentation
            mask, detected_objects = self.segmenter.segment_image(image)
            context['segmentation_results'] = detected_objects
            
            # Determine scene type based on detected objects
            context['scene_type'] = self._determine_scene_type(detected_objects)
            
            # Extract dominant objects
            if detected_objects['dominant_classes']:
                context['dominant_objects'] = [
                    obj['class_name'] for obj in detected_objects['dominant_classes'][:5]
                ]
            
            # Analyze object relationships
            context['object_relationships'] = self._analyze_object_relationships(
                detected_objects, context['scene_type']
            )
            
            # Estimate lighting conditions
            context['lighting_estimate'] = self._estimate_lighting(image, mask)
            
            # Get color harmony suggestions
            context['color_harmony_suggestions'] = self._get_color_harmony_suggestions(
                context['scene_type']
            )
            
            return context
            
        except Exception as e:
            print(f"⚠️ Scene context analysis error: {e}")
            context['error'] = str(e)
            return context
    
    def _determine_scene_type(self, detected_objects: Dict) -> str:
        """
        Determine the type of scene based on detected objects.
        
        Args:
            detected_objects: Segmentation results
        
        Returns:
            Scene type string
        """
        object_counts = detected_objects.get('class_counts', {})
        
        # Count objects by category
        category_counts = {
            'outdoor': 0,
            'urban': 0,
            'nature': 0,
            'indoor': 0
        }
        
        for class_name in object_counts.keys():
            if class_name in ['sky', 'tree', 'grass', 'mountain', 'sea']:
                category_counts['outdoor'] += object_counts[class_name]
                category_counts['nature'] += object_counts[class_name]
            elif class_name in ['building', 'car', 'road', 'traffic light', 'bus']:
                category_counts['urban'] += object_counts[class_name]
                category_counts['outdoor'] += object_counts[class_name]
            elif class_name in ['chair', 'couch', 'bed', 'dining table', 'tv']:
                category_counts['indoor'] += object_counts[class_name]
        
        # Return scene type with highest count
        return max(category_counts, key=category_counts.get)
    
    def _analyze_object_relationships(self, detected_objects: Dict, scene_type: str) -> List[Dict]:
        """
        Analyze relationships between objects in the scene.
        
        Args:
            detected_objects: Segmentation results
            scene_type: Type of scene
        
        Returns:
            List of object relationship descriptions
        """
        relationships = []
        
        # Get dominant objects
        dominant = detected_objects.get('dominant_classes', [])
        if not dominant:
            return relationships
        
        # Common object relationships based on scene type
        if scene_type == 'outdoor':
            relationships.append({
                'type': 'sky_ground',
                'description': 'Sky and ground relationship',
                'importance': 'high'
            })
            relationships.append({
                'type': 'lighting_consistency',
                'description': 'Consistent lighting across objects',
                'importance': 'high'
            })
        
        elif scene_type == 'urban':
            relationships.append({
                'type': 'building_road',
                'description': 'Building and road color harmony',
                'importance': 'medium'
            })
            relationships.append({
                'type': 'vehicle_integration',
                'description': 'Vehicle colors in urban context',
                'importance': 'medium'
            })
        
        elif scene_type == 'nature':
            relationships.append({
                'type': 'vegetation_water',
                'description': 'Vegetation and water color interaction',
                'importance': 'high'
            })
            relationships.append({
                'type': 'natural_lighting',
                'description': 'Natural lighting conditions',
                'importance': 'high'
            })
        
        return relationships
    
    def _estimate_lighting(self, image: Image.Image, mask: np.ndarray) -> str:
        """
        Estimate lighting conditions from the image.
        
        Args:
            image: Input image
            mask: Segmentation mask
        
        Returns:
            Lighting condition string
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate overall brightness
        if len(img_array.shape) == 3:
            brightness = np.mean(img_array)
        else:
            brightness = np.mean(img_array)
        
        # Classify lighting
        if brightness < 80:
            return 'low_light'
        elif brightness < 150:
            return 'normal'
        elif brightness < 200:
            return 'bright'
        else:
            return 'very_bright'
    
    def _get_color_harmony_suggestions(self, scene_type: str) -> Dict:
        """
        Get color harmony suggestions for the scene type.
        
        Args:
            scene_type: Type of scene
        
        Returns:
            Dictionary of color harmony suggestions
        """
        return self.COLOR_HARMONY_PALETTES.get(scene_type, self.COLOR_HARMONY_PALETTES['outdoor'])
    
    def apply_context_aware_colorization(
        self,
        image: Image.Image,
        context: Optional[Dict] = None,
        strength: float = 0.8
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply context-aware colorization considering scene relationships.
        
        Args:
            image: Input PIL Image
            context: Scene context (if None, will be analyzed)
            strength: Colorization strength (0.0-1.0)
        
        Returns:
            Tuple of (colorized_image, metadata_dict)
        """
        metadata = {
            'method': 'context_aware_colorization',
            'strength': strength
        }
        
        try:
            # Analyze scene context if not provided
            if context is None:
                context = self.analyze_scene_context(image)
            
            metadata['scene_context'] = context
            
            # Create context-aware prompt
            prompt = self._create_context_aware_prompt(context)
            
            # Apply base colorization with context-aware prompt
            colorized, color_metadata = self.colorizer.colorize(
                image,
                prompt=prompt,
                strength=strength
            )
            
            metadata['colorization_metadata'] = color_metadata
            metadata['prompt_used'] = prompt
            
            # Apply context-specific refinements
            refined = self._apply_context_refinements(colorized, context)
            
            return refined, metadata
            
        except Exception as e:
            print(f"⚠️ Context-aware colorization error: {e}")
            metadata['error'] = str(e)
            # Fallback to basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
    
    def _create_context_aware_prompt(self, context: Dict) -> str:
        """
        Create a colorization prompt based on scene context.
        
        Args:
            context: Scene context information
        
        Returns:
            Context-aware colorization prompt
        """
        base_prompt = "colorize this black and white photo, realistic colors, high quality"
        
        scene_type = context.get('scene_type', 'outdoor')
        lighting = context.get('lighting_estimate', 'normal')
        
        # Add scene-specific instructions
        scene_prompts = {
            'outdoor': "natural outdoor lighting, consistent colors between sky and ground",
            'urban': "urban scene with realistic building and road colors, proper vehicle colors",
            'nature': "natural scene with realistic vegetation and water colors, natural lighting",
            'indoor': "indoor scene with consistent wall and furniture colors, proper lighting"
        }
        
        # Add lighting-specific instructions
        lighting_prompts = {
            'low_light': "low light conditions, darker tones",
            'normal': "normal lighting conditions",
            'bright': "bright lighting conditions, vibrant colors",
            'very_bright': "very bright lighting, high contrast"
        }
        
        scene_instruction = scene_prompts.get(scene_type, "")
        lighting_instruction = lighting_prompts.get(lighting, "")
        
        # Combine prompts
        if scene_instruction and lighting_instruction:
            enhanced_prompt = f"{base_prompt}, {scene_instruction}, {lighting_instruction}"
        elif scene_instruction:
            enhanced_prompt = f"{base_prompt}, {scene_instruction}"
        elif lighting_instruction:
            enhanced_prompt = f"{base_prompt}, {lighting_instruction}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt
    
    def _apply_context_refinements(self, image: Image.Image, context: Dict) -> Image.Image:
        """
        Apply context-specific refinements to the colorized image.
        
        Args:
            image: Colorized image
            context: Scene context information
        
        Returns:
            Refined image
        """
        refined = image.copy()
        
        try:
            scene_type = context.get('scene_type', 'outdoor')
            lighting = context.get('lighting_estimate', 'normal')
            
            # Apply scene-specific enhancements
            if scene_type == 'outdoor':
                # Enhance natural colors
                refined = self.image_processor.adjust_saturation(refined, 1.1)
                refined = self.image_processor.adjust_contrast(refined, 1.05)
            
            elif scene_type == 'urban':
                # Enhance structural clarity
                refined = self.image_processor.sharpen_image(refined, 1.1)
                refined = self.image_processor.adjust_contrast(refined, 1.1)
            
            elif scene_type == 'nature':
                # Enhance natural tones
                refined = self.image_processor.adjust_saturation(refined, 1.15)
                refined = self.image_processor.adjust_brightness(refined, 1.05)
            
            # Apply lighting-specific adjustments
            if lighting == 'low_light':
                refined = self.image_processor.adjust_brightness(refined, 1.2)
            elif lighting == 'very_bright':
                refined = self.image_processor.adjust_brightness(refined, 0.9)
            
            return refined
            
        except Exception as e:
            print(f"⚠️ Context refinement error: {e}")
            return image
    
    def create_context_visualization(
        self,
        original: Image.Image,
        context: Dict
    ) -> Image.Image:
        """
        Create a visualization showing the analyzed scene context.
        
        Args:
            original: Original image
            context: Scene context information
        
        Returns:
            Visualization image with context annotations
        """
        try:
            # Create a copy for annotation
            viz = original.copy()
            
            # If we have segmentation results, create overlay
            if 'segmentation_results' in context:
                seg_results = context['segmentation_results']
                mask, _ = self.segmenter.segment_image(original)
                overlay = self.segmenter.create_segmentation_overlay(original, mask, alpha=0.3)
                viz = overlay
            
            return viz
            
        except Exception as e:
            print(f"⚠️ Context visualization error: {e}")
            return original


if __name__ == "__main__":
    # Test context colorizer
    print("Testing Context-Aware Colorizer...")
    
    try:
        colorizer = ContextColorizer()
        
        # Test with image
        test_img = Image.new('RGB', (512, 512), color='gray')
        
        # Test scene context analysis
        context = colorizer.analyze_scene_context(test_img)
        print(f"✅ Scene context analyzed: {context['scene_type']}")
        print(f"✅ Dominant objects: {context['dominant_objects']}")
        
        # Test context-aware colorization
        colorized, metadata = colorizer.apply_context_aware_colorization(test_img)
        print(f"✅ Context-aware colorization complete! Metadata keys: {metadata.keys()}")
        
        # Test context visualization
        viz = colorizer.create_context_visualization(test_img, context)
        print("✅ Context visualization complete!")
        
        print("✅ Context-aware colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")