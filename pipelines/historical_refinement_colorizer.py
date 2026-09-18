"""
Historical Photograph Refinement Colorization Pipeline for Task 6
Deep refinement and period-specific treatment for historical photographs.
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List

from pipelines.base_colorizer import BaseColorizer
from historical.era_detector import EraDetector
from historical.historical_palette import HistoricalPalette
from historical.historical_refinement import HistoricalRefinement
from processing.image_processor import ImageProcessor


class HistoricalRefinementColorizer:
    """
    Advanced historical photograph colorization with deep refinement.
    Builds on Task 4's historical infrastructure with period-specific treatments.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize historical refinement colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"🖼️ Initializing Historical Refinement Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize era detector (from Task 4)
            self.era_detector = EraDetector()
            
            # Initialize historical palette (from Task 4)
            self.palette = HistoricalPalette()
            
            # Initialize historical refinement (Task 6 specific)
            self.refinement = HistoricalRefinement()
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            print("✅ Historical Refinement Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize historical refinement colorizer: {e}")
            raise
    
    def colorize_historical_refined(
        self,
        image: Image.Image,
        period: Optional[str] = None,
        auto_detect_period: bool = True,
        refinement_intensity: float = 0.8,
        colorization_strength: float = 0.8,
        preserve_facial_features: bool = True,
        era_treatment_intensity: float = 0.7
    ) -> Tuple[Image.Image, Dict]:
        """
        Colorize a historical photograph with deep period-specific refinement.
        
        Args:
            image: Input historical photograph
            period: Specific historical period (None = auto-detect)
            auto_detect_period: Whether to auto-detect period if not specified
            refinement_intensity: Intensity of refinement treatments (0.0-1.0)
            colorization_strength: Strength of base colorization (0.0-1.0)
            preserve_facial_features: Whether to preserve facial feature details
            era_treatment_intensity: Intensity of era-specific color treatment (0.0-1.0)
        
        Returns:
            Tuple of (refined_colorized_image, metadata_dict)
        """
        metadata = {
            'method': 'historical_refinement_colorization',
            'auto_detect_period': auto_detect_period,
            'refinement_intensity': refinement_intensity,
            'colorization_strength': colorization_strength,
            'preserve_facial_features': preserve_facial_features,
            'era_treatment_intensity': era_treatment_intensity
        }
        
        try:
            # Determine historical period
            if period:
                if period not in self.refinement.get_available_periods():
                    raise ValueError(f"Invalid period: {period}. Available: {self.refinement.get_available_periods()}")
                selected_period = period
                metadata['period_selection'] = 'manual'
            elif auto_detect_period:
                # Use era detection from Task 4, map to refinement periods
                era_result = self.era_detector.detect_era(image)
                selected_period = self._map_era_to_period(era_result['detected_era'])
                metadata['period_selection'] = 'auto_detected'
                metadata['era_detection'] = era_result
            else:
                selected_period = '1950s'  # Default
                metadata['period_selection'] = 'default'
            
            metadata['selected_period'] = selected_period
            
            # Step 1: Base colorization with era-specific prompt
            era_prompt = self._create_period_specific_prompt(selected_period)
            colorized, color_metadata = self.colorizer.colorize(
                image,
                prompt=era_prompt,
                strength=colorization_strength
            )
            metadata['base_colorization'] = color_metadata
            metadata['period_prompt_used'] = era_prompt
            
            # Step 2: Apply era-specific color treatment (from Task 4)
            era_for_treatment = self._map_period_to_era(selected_period)
            treated, treatment_metadata = self.palette.apply_era_color_treatment(
                colorized,
                era_for_treatment,
                intensity=era_treatment_intensity
            )
            metadata['era_treatment'] = treatment_metadata
            
            # Step 3: Apply period-specific refinement (Task 6)
            refined, refinement_metadata = self.refinement.refine_historical_photo(
                treated,
                period=selected_period,
                refinement_intensity=refinement_intensity,
                preserve_facial_features=preserve_facial_features
            )
            metadata['period_refinement'] = refinement_metadata
            
            # Step 4: Final historical enhancement
            final = self.image_processor.enhance_for_historical(refined, era_for_treatment)
            metadata['final_enhancement'] = True
            
            return final, metadata
            
        except Exception as e:
            print(f"⚠️ Historical refinement colorization error: {e}")
            metadata['error'] = str(e)
            # Fallback to basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
    
    def _map_era_to_period(self, era: str) -> str:
        """
        Map detected era to refinement period.
        
        Args:
            era: Detected era string
        
        Returns:
            Refinement period string
        """
        era_to_period_mapping = {
            '1900s': '1920s',      # Map early eras to 1920s
            '1910s': '1920s',
            '1920s': '1920s',
            '1930s': '1920s',
            '1940s': 'WWII',
            '1950s': '1950s',
            '1960s': '1960s',
            '1970s': '1960s'       # Map later eras to 1960s
        }
        
        return era_to_period_mapping.get(era, '1950s')
    
    def _map_period_to_era(self, period: str) -> str:
        """
        Map refinement period back to era for color treatment.
        
        Args:
            period: Refinement period string
        
        Returns:
            Era string for palette treatment
        """
        period_to_era_mapping = {
            '1920s': '1920s',
            'WWII': '1940s',
            '1950s': '1950s',
            '1960s': '1960s'
        }
        
        return period_to_era_mapping.get(period, '1950s')
    
    def _create_period_specific_prompt(self, period: str) -> str:
        """
        Create a colorization prompt specific to the historical period.
        
        Args:
            period: Historical period string
        
        Returns:
            Period-specific colorization prompt
        """
        base_prompt = "colorize this historical black and white photograph, realistic colors, high quality"
        
        period_prompts = {
            '1920s': "1920s Art Deco style, vintage colors, period-appropriate aesthetics",
            'WWII': "WWII documentary style, authentic military colors, historical accuracy",
            '1950s': "1950s Kodachrome style, vibrant but realistic colors, post-war aesthetic",
            '1960s': "1960s cultural revolution style, natural colors, vintage film look"
        }
        
        period_instruction = period_prompts.get(period, "historical photograph style")
        
        return f"{base_prompt}, {period_instruction}"
    
    def batch_refine_historical(
        self,
        images: list,
        period: Optional[str] = None,
        auto_detect_period: bool = True,
        **kwargs
    ) -> list:
        """
        Colorize and refine multiple historical images with the same period settings.
        
        Args:
            images: List of PIL Images
            period: Period to use for all images
            auto_detect_period: Whether to auto-detect period
            **kwargs: Additional arguments for colorize_historical_refined
        
        Returns:
            List of (refined_colorized_image, metadata) tuples
        """
        results = []
        for i, img in enumerate(images):
            print(f"Processing historical image {i+1}/{len(images)} with refinement")
            refined, metadata = self.colorize_historical_refined(
                img, period, auto_detect_period, **kwargs
            )
            results.append((refined, metadata))
        return results
    
    def compare_refinement_periods(
        self,
        image: Image.Image,
        periods: Optional[List[str]] = None
    ) -> Dict[str, Tuple[Image.Image, Dict]]:
        """
        Compare refinement results across different historical periods.
        
        Args:
            image: Input image
            periods: List of periods to compare (None = use all available)
        
        Returns:
            Dictionary mapping period names to (refined_image, metadata) tuples
        """
        if periods is None:
            periods = self.refinement.get_available_periods()
        
        comparison_results = {}
        
        for period in periods:
            try:
                refined, metadata = self.colorize_historical_refined(
                    image, period=period, auto_detect_period=False
                )
                comparison_results[period] = (refined, metadata)
            except Exception as e:
                print(f"⚠️ Error processing period {period}: {e}")
                comparison_results[period] = (image, {'error': str(e)})
        
        return comparison_results
    
    def get_refinement_pipeline_info(self) -> Dict:
        """
        Get information about the refinement pipeline stages.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'pipeline_stages': [
                'Base colorization with period-specific prompt',
                'Era-specific color treatment (Task 4 infrastructure)',
                'Period-specific refinement (Task 6)',
                'Final historical enhancement'
            ],
            'integration_with_task_4': 'Uses era detection and palette from Task 4',
            'task_6_specific': 'Adds deep period-specific refinement and facial preservation',
            'available_periods': self.refinement.get_available_periods()
        }
    
    def get_available_periods(self) -> List[str]:
        """
        Get list of available historical periods for refinement.
        
        Returns:
            List of period names
        """
        return self.refinement.get_available_periods()
    
    def get_period_info(self, period: str) -> Optional[Dict]:
        """
        Get detailed information about a specific period.
        
        Args:
            period: Period string
        
        Returns:
            Period information dictionary or None if invalid
        """
        return self.refinement.get_period_info(period)


if __name__ == "__main__":
    # Test historical refinement colorizer
    print("Testing Historical Refinement Colorizer...")
    
    try:
        colorizer = HistoricalRefinementColorizer()
        
        # Test with image
        test_img = Image.new('RGB', (512, 512), color='gray')
        
        # Test historical refinement colorization with auto-detection
        refined, metadata = colorizer.colorize_historical_refined(test_img)
        print(f"✅ Historical refinement colorization complete! Period: {metadata['selected_period']}")
        print(f"✅ Pipeline stages: {len([k for k in metadata.keys() if not k.startswith('_')])}")
        
        # Test with manual period selection
        refined_manual, metadata_manual = colorizer.colorize_historical_refined(
            test_img, period='1950s', auto_detect_period=False
        )
        print(f"✅ Manual period colorization complete! Period: {metadata_manual['selected_period']}")
        
        # Test available periods
        periods = colorizer.get_available_periods()
        print(f"✅ Available periods: {periods}")
        
        # Test period comparison
        comparison = colorizer.compare_refinement_periods(test_img, ['1920s', '1950s'])
        print(f"✅ Period comparison complete! Compared {len(comparison)} periods")
        
        # Test pipeline info
        pipeline_info = colorizer.get_refinement_pipeline_info()
        print(f"✅ Pipeline stages: {len(pipeline_info['pipeline_stages'])}")
        
        print("✅ Historical refinement colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")