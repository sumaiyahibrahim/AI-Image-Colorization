"""
Historical Image Colorization Pipeline for Task 4
Time-based colorization with era detection and manual override.

Author: Sumaiya Ibrahim
Task: Task 4 - Time-Based Historical Image Colorization
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

from pipelines.base_colorizer import BaseColorizer
from historical.era_detector import EraDetector
from historical.historical_palette import HistoricalPalette
from processing.image_processor import ImageProcessor


class HistoricalColorizer:
    """
    Historical image colorization with era detection and era-specific color treatments.
    Supports automatic era recognition and manual override.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize historical colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"📅 Initializing Historical Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize era detector
            self.era_detector = EraDetector()
            
            # Initialize historical palette
            self.palette = HistoricalPalette()
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            print("✅ Historical Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize historical colorizer: {e}")
            raise
    
    def colorize_historical_image(
        self,
        image: Image.Image,
        era: Optional[str] = None,
        manual_override: bool = False,
        treatment_intensity: float = 0.7,
        colorization_strength: float = 0.8
    ) -> Tuple[Image.Image, Dict]:
        """
        Colorize a historical photograph with era-specific treatment.
        
        Args:
            image: Input PIL Image (historical photograph)
            era: Specific era to use (None = auto-detect)
            manual_override: Whether to force manual era selection
            treatment_intensity: Intensity of era-specific treatment (0.0-1.0)
            colorization_strength: Strength of base colorization (0.0-1.0)
        
        Returns:
            Tuple of (colorized_image, metadata_dict)
        """
        metadata = {
            'method': 'historical_colorization',
            'manual_override': manual_override,
            'treatment_intensity': treatment_intensity,
            'colorization_strength': colorization_strength
        }
        
        try:
            # Determine era
            if era and manual_override:
                # Use manual era override
                override_result = self.era_detector.manual_override(era)
                if not override_result['success']:
                    raise ValueError(override_result['error'])
                
                selected_era = era
                metadata['era_selection'] = 'manual_override'
                metadata['era_override_info'] = override_result
            else:
                # Auto-detect era
                detection_result = self.era_detector.detect_era(image)
                selected_era = detection_result['detected_era']
                metadata['era_selection'] = 'auto_detected'
                metadata['era_detection'] = detection_result
                
                # If manual override was requested but no era provided, use detection
                if manual_override and not era:
                    selected_era = detection_result['detected_era']
                    metadata['era_selection'] = 'auto_fallback'
            
            metadata['selected_era'] = selected_era
            
            # Create era-specific colorization prompt
            era_prompt = self._create_era_specific_prompt(selected_era)
            
            # Apply base colorization with era-specific prompt
            colorized, color_metadata = self.colorizer.colorize(
                image,
                prompt=era_prompt,
                strength=colorization_strength
            )
            
            metadata['colorization_metadata'] = color_metadata
            metadata['era_prompt_used'] = era_prompt
            
            # Apply era-specific color treatment
            treated, treatment_metadata = self.palette.apply_era_color_treatment(
                colorized,
                selected_era,
                intensity=treatment_intensity
            )
            
            metadata['treatment_metadata'] = treatment_metadata
            
            # Apply historical-specific enhancements
            enhanced = self.image_processor.enhance_for_historical(treated, selected_era)
            metadata['historical_enhancements'] = True
            
            return enhanced, metadata
            
        except Exception as e:
            print(f"⚠️ Historical colorization error: {e}")
            metadata['error'] = str(e)
            # Fallback to basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
    
    def _create_era_specific_prompt(self, era: str) -> str:
        """
        Create a colorization prompt specific to the historical era.
        
        Args:
            era: Historical era string
        
        Returns:
            Era-specific colorization prompt
        """
        base_prompt = "colorize this historical black and white photograph, realistic colors, high quality"
        
        era_prompts = {
            '1900s': "Victorian era style, sepia undertones, early 1900s aesthetic",
            '1910s': "WWI era photography style, muted colors, documentary feel",
            '1920s': "Roaring Twenties style, Art Deco colors, vintage aesthetic",
            '1930s': "Great Depression era style, muted tones, documentary photography",
            '1940s': "WWII era photography, military style, documentary colors",
            '1950s': "Post-war prosperity style, vibrant but realistic colors, 1950s aesthetic",
            '1960s': "1960s cultural revolution style, natural colors, vintage film look",
            '1970s': "1970s vintage film style, warm colors, film grain aesthetic"
        }
        
        era_instruction = era_prompts.get(era, "historical photograph style")
        
        return f"{base_prompt}, {era_instruction}"
    
    def batch_colorize_historical(
        self,
        images: list,
        era: Optional[str] = None,
        manual_override: bool = False,
        **kwargs
    ) -> list:
        """
        Colorize multiple historical images with the same era settings.
        
        Args:
            images: List of PIL Images
            era: Era to use for all images
            manual_override: Whether to use manual era selection
            **kwargs: Additional arguments for colorize_historical_image
        
        Returns:
            List of (colorized_image, metadata) tuples
        """
        results = []
        for i, img in enumerate(images):
            print(f"Processing historical image {i+1}/{len(images)}")
            colorized, metadata = self.colorize_historical_image(
                img, era, manual_override, **kwargs
            )
            results.append((colorized, metadata))
        return results
    
    def compare_eras(
        self,
        image: Image.Image,
        eras: Optional[list] = None
    ) -> Dict[str, Tuple[Image.Image, Dict]]:
        """
        Compare colorization results across different historical eras.
        
        Args:
            image: Input image
            eras: List of eras to compare (None = use all available eras)
        
        Returns:
            Dictionary mapping era names to (colorized_image, metadata) tuples
        """
        if eras is None:
            eras = self.era_detector.get_available_eras()
        
        comparison_results = {}
        
        for era in eras:
            try:
                colorized, metadata = self.colorize_historical_image(
                    image,
                    era=era,
                    manual_override=True
                )
                comparison_results[era] = (colorized, metadata)
            except Exception as e:
                print(f"⚠️ Error processing era {era}: {e}")
                comparison_results[era] = (image, {'error': str(e)})
        
        return comparison_results
    
    def get_era_detection_info(self, image: Image.Image) -> Dict:
        """
        Get detailed era detection information for an image.
        
        Args:
            image: Input image
        
        Returns:
            Detailed era detection information
        """
        return self.era_detector.detect_era(image)
    
    def get_available_eras(self) -> list:
        """
        Get list of available historical eras.
        
        Returns:
            List of era names
        """
        return self.era_detector.get_available_eras()
    
    def get_era_palette_preview(self, era: str) -> Image.Image:
        """
        Get a color palette preview for a specific era.
        
        Args:
            era: Era string
        
        Returns:
            Palette preview image
        """
        return self.palette.create_palette_preview(era)


if __name__ == "__main__":
    # Test historical colorizer
    print("Testing Historical Colorizer...")
    
    try:
        colorizer = HistoricalColorizer()
        
        # Test with image
        test_img = Image.new('RGB', (512, 512), color='gray')
        
        # Test era detection
        era_info = colorizer.get_era_detection_info(test_img)
        print(f"✅ Detected era: {era_info['detected_era']}")
        print(f"✅ Confidence: {era_info['confidence']:.2f}")
        
        # Test historical colorization with auto-detection
        colorized, metadata = colorizer.colorize_historical_image(test_img)
        print(f"✅ Historical colorization complete! Selected era: {metadata['selected_era']}")
        
        # Test with manual override
        colorized_manual, metadata_manual = colorizer.colorize_historical_image(
            test_img, era='1920s', manual_override=True
        )
        print(f"✅ Manual override colorization complete! Era: {metadata_manual['selected_era']}")
        
        # Test available eras
        eras = colorizer.get_available_eras()
        print(f"✅ Available eras: {eras}")
        
        # Test era comparison
        comparison = colorizer.compare_eras(test_img, ['1920s', '1950s'])
        print(f"✅ Era comparison complete! Compared {len(comparison)} eras")
        
        print("✅ Historical colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")