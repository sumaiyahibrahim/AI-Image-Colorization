"""
Historical Photograph Refinement Module for Task 6
Deep refinement and period-specific treatment for historical photographs.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Tuple, Optional, List
import cv2

from processing.image_processor import ImageProcessor
from historical.historical_palette import HistoricalPalette


class HistoricalRefinement:
    """
    Advanced refinement for historical photograph colorization.
    Provides period-specific treatments and quality enhancements.
    """
    
    # Period-specific refinement parameters
    PERIOD_REFINEMENTS = {
        '1920s': {
            'name': '1920s Art Deco Period',
            'color_adjustments': {
                'warmth_boost': 0.15,
                'saturation_target': 0.35,
                'contrast_enhancement': 1.05,
                'shadow_tint': (20, 10, 0),  # Warm shadows
                'highlight_tint': (240, 240, 255)  # Cool highlights
            },
            'texture_enhancement': {
                'grain_reduction': 0.3,
                'edge_preservation': 0.8,
                'detail_enhancement': 1.1
            },
            'artistic_treatment': {
                'vignette_intensity': 0.2,
                'soft_focus': 0.1,
                'period_glow': 0.05
            }
        },
        'WWII': {
            'name': 'WWII Period (1939-1945)',
            'color_adjustments': {
                'warmth_boost': 0.05,
                'saturation_target': 0.30,
                'contrast_enhancement': 1.10,
                'shadow_tint': (10, 5, 5),  # Slightly warm shadows
                'highlight_tint': (245, 245, 250)  # Neutral highlights
            },
            'texture_enhancement': {
                'grain_reduction': 0.4,
                'edge_preservation': 0.9,
                'detail_enhancement': 1.15
            },
            'artistic_treatment': {
                'vignette_intensity': 0.15,
                'soft_focus': 0.05,
                'documentary_feel': 0.1
            }
        },
        '1950s': {
            'name': '1950s Post-War Era',
            'color_adjustments': {
                'warmth_boost': 0.10,
                'saturation_target': 0.45,
                'contrast_enhancement': 1.08,
                'shadow_tint': (15, 10, 5),  # Warm shadows
                'highlight_tint': (250, 248, 245)  # Warm highlights
            },
            'texture_enhancement': {
                'grain_reduction': 0.5,
                'edge_preservation': 0.85,
                'detail_enhancement': 1.2
            },
            'artistic_treatment': {
                'vignette_intensity': 0.1,
                'soft_focus': 0.0,
                'kodachrome_style': 0.15
            }
        },
        '1960s': {
            'name': '1960s Cultural Revolution',
            'color_adjustments': {
                'warmth_boost': 0.05,
                'saturation_target': 0.55,
                'contrast_enhancement': 1.12,
                'shadow_tint': (10, 10, 15),  # Cool shadows
                'highlight_tint': (255, 253, 250)  # Warm highlights
            },
            'texture_enhancement': {
                'grain_reduction': 0.6,
                'edge_preservation': 0.9,
                'detail_enhancement': 1.25
            },
            'artistic_treatment': {
                'vignette_intensity': 0.05,
                'soft_focus': 0.0,
                'vibrant_boost': 0.2
            }
        }
    }
    
    def __init__(self):
        """Initialize the historical refinement system."""
        print(f"🖼️ Initializing Historical Photograph Refinement System")
        
        # Initialize image processor
        self.image_processor = ImageProcessor()
        
        # Initialize historical palette
        self.palette = HistoricalPalette()
        
        print("✅ Historical Refinement System Ready!")
    
    def refine_historical_photo(
        self,
        image: Image.Image,
        period: str = '1950s',
        refinement_intensity: float = 0.8,
        preserve_facial_features: bool = True
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply advanced refinement to a historical photograph.
        
        Args:
            image: Input colorized historical photograph
            period: Historical period for specific treatment
            refinement_intensity: Intensity of refinements (0.0-1.0)
            preserve_facial_features: Whether to preserve facial feature details
        
        Returns:
            Tuple of (refined_image, metadata_dict)
        """
        metadata = {
            'period': period,
            'refinement_intensity': refinement_intensity,
            'preserve_facial_features': preserve_facial_features,
            'method': 'historical_refinement'
        }
        
        if period not in self.PERIOD_REFINEMENTS:
            metadata['error'] = f'Period {period} not found'
            return image, metadata
        
        try:
            period_params = self.PERIOD_REFINEMENTS[period]
            metadata['period_params'] = period_params
            
            refined = image.copy()
            
            # Apply period-specific color adjustments
            refined = self._apply_period_color_adjustments(
                refined, period_params['color_adjustments'], refinement_intensity
            )
            metadata['color_adjustments_applied'] = True
            
            # Apply texture enhancement
            refined = self._apply_texture_enhancement(
                refined, period_params['texture_enhancement'], refinement_intensity
            )
            metadata['texture_enhancement_applied'] = True
            
            # Apply artistic treatment
            refined = self._apply_artistic_treatment(
                refined, period_params['artistic_treatment'], refinement_intensity
            )
            metadata['artistic_treatment_applied'] = True
            
            # Facial feature preservation if requested
            if preserve_facial_features:
                refined = self._preserve_facial_features(refined, image)
                metadata['facial_preservation_applied'] = True
            
            # Final quality enhancement
            refined = self._final_quality_enhancement(refined)
            metadata['final_enhancement_applied'] = True
            
            return refined, metadata
            
        except Exception as e:
            print(f"⚠️ Historical refinement error: {e}")
            metadata['error'] = str(e)
            return image, metadata
    
    def _apply_period_color_adjustments(
        self,
        image: Image.Image,
        color_params: Dict,
        intensity: float
    ) -> Image.Image:
        """
        Apply period-specific color adjustments.
        
        Args:
            image: Input image
            color_params: Color adjustment parameters
            intensity: Adjustment intensity
        
        Returns:
            Color-adjusted image
        """
        adjusted = image.copy()
        
        try:
            # Warmth boost
            warmth_boost = color_params.get('warmth_boost', 0) * intensity
            if warmth_boost > 0:
                # Increase red channel slightly
                r, g, b = adjusted.split()
                r_enhanced = ImageEnhance.Brightness(r).enhance(1 + warmth_boost * 0.1)
                adjusted = Image.merge('RGB', (r_enhanced, g, b))
            
            # Saturation target
            sat_target = color_params.get('saturation_target', 0.5)
            if sat_target != 0.5:
                current_sat = self._estimate_saturation(adjusted)
                sat_factor = (sat_target / (current_sat + 0.01)) * intensity
                sat_factor = np.clip(sat_factor, 0.5, 1.5)
                adjusted = self.image_processor.adjust_saturation(adjusted, sat_factor)
            
            # Contrast enhancement
            contrast_enh = color_params.get('contrast_enhancement', 1.0)
            if contrast_enh != 1.0:
                contrast_factor = 1 + (contrast_enh - 1) * intensity
                adjusted = self.image_processor.adjust_contrast(adjusted, contrast_factor)
            
            # Shadow and highlight tints
            shadow_tint = color_params.get('shadow_tint')
            highlight_tint = color_params.get('highlight_tint')
            
            if shadow_tint or highlight_tint:
                adjusted = self._apply_shadow_highlight_tints(
                    adjusted, shadow_tint, highlight_tint, intensity
                )
            
            return adjusted
            
        except Exception as e:
            print(f"⚠️ Color adjustment error: {e}")
            return image
    
    def _estimate_saturation(self, image: Image.Image) -> float:
        """Estimate average saturation of an image."""
        hsv = image.convert('HSV')
        s = hsv.split()[1]
        return np.mean(np.array(s)) / 255.0
    
    def _apply_shadow_highlight_tints(
        self,
        image: Image.Image,
        shadow_tint: Optional[Tuple[int, int, int]],
        highlight_tint: Optional[Tuple[int, int, int]],
        intensity: float
    ) -> Image.Image:
        """Apply tints to shadows and highlights."""
        if not shadow_tint and not highlight_tint:
            return image
        
        img_array = np.array(image).astype(np.float32)
        
        # Create luminance mask
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        
        # Shadow mask (dark regions)
        shadow_mask = (luminance < 128).astype(np.float32) * intensity
        
        # Highlight mask (bright regions)
        highlight_mask = (luminance >= 128).astype(np.float32) * intensity
        
        # Apply shadow tint
        if shadow_tint:
            shadow_color = np.array(shadow_tint) / 255.0
            for i in range(3):
                img_array[:, :, i] = (
                    img_array[:, :, i] * (1 - shadow_mask) + 
                    shadow_color[i] * shadow_mask
                )
        
        # Apply highlight tint
        if highlight_tint:
            highlight_color = np.array(highlight_tint) / 255.0
            for i in range(3):
                img_array[:, :, i] = (
                    img_array[:, :, i] * (1 - highlight_mask) + 
                    highlight_color[i] * highlight_mask
                )
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _apply_texture_enhancement(
        self,
        image: Image.Image,
        texture_params: Dict,
        intensity: float
    ) -> Image.Image:
        """
        Apply texture enhancement.
        
        Args:
            image: Input image
            texture_params: Texture enhancement parameters
            intensity: Enhancement intensity
        
        Returns:
            Texture-enhanced image
        """
        enhanced = image.copy()
        
        try:
            # Grain reduction
            grain_reduction = texture_params.get('grain_reduction', 0) * intensity
            if grain_reduction > 0:
                enhanced = self.image_processor.denoise_image(enhanced, strength=grain_reduction * 2)
            
            # Edge preservation and detail enhancement
            edge_preservation = texture_params.get('edge_preservation', 0.8)
            detail_enhancement = texture_params.get('detail_enhancement', 1.0)
            
            if detail_enhancement > 1.0:
                # Apply subtle sharpening
                sharpen_factor = 1 + (detail_enhancement - 1) * intensity
                enhanced = self.image_processor.sharpen_image(enhanced, sharpen_factor)
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ Texture enhancement error: {e}")
            return image
    
    def _apply_artistic_treatment(
        self,
        image: Image.Image,
        artistic_params: Dict,
        intensity: float
    ) -> Image.Image:
        """
        Apply period-specific artistic treatments.
        
        Args:
            image: Input image
            artistic_params: Artistic treatment parameters
            intensity: Treatment intensity
        
        Returns:
            Artistically treated image
        """
        treated = image.copy()
        
        try:
            # Vignette
            vignette_intensity = artistic_params.get('vignette_intensity', 0) * intensity
            if vignette_intensity > 0:
                treated = self.image_processor.vignette_effect(treated, vignette_intensity)
            
            # Soft focus
            soft_focus = artistic_params.get('soft_focus', 0) * intensity
            if soft_focus > 0:
                treated = treated.filter(ImageFilter.GaussianBlur(radius=soft_focus * 2))
            
            # Period-specific treatments
            if 'kodachrome_style' in artistic_params:
                kodachrome = artistic_params['kodachrome_style'] * intensity
                if kodachrome > 0:
                    treated = self.image_processor.adjust_saturation(treated, 1 + kodachrome * 0.3)
                    treated = self.image_processor.adjust_contrast(treated, 1 + kodachrome * 0.1)
            
            if 'vibrant_boost' in artistic_params:
                vibrant = artistic_params['vibrant_boost'] * intensity
                if vibrant > 0:
                    treated = self.image_processor.adjust_saturation(treated, 1 + vibrant * 0.4)
            
            return treated
            
        except Exception as e:
            print(f"⚠️ Artistic treatment error: {e}")
            return image
    
    def _preserve_facial_features(self, refined: Image.Image, original: Image.Image) -> Image.Image:
        """
        Preserve facial features from the original image.
        
        Args:
            refined: Refined image
            original: Original image
        
        Returns:
            Image with preserved facial features
        """
        try:
            # Use a simple approach: blend facial regions more conservatively
            # In a full implementation, this would use face detection
            
            # For now, apply a conservative blend in the center region
            # (where faces are typically located)
            width, height = refined.size
            
            # Define center region (typically contains faces)
            center_x = width // 2
            center_y = height // 2
            region_size = min(width, height) // 3
            
            # Create a gradient mask for smooth blending
            mask = Image.new('L', (width, height), 0)
            mask_array = np.array(mask)
            
            # Create radial gradient for center region
            y, x = np.ogrid[:height, :width]
            center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask_array = np.clip(1 - (center_dist / region_size), 0, 1) * 128
            
            mask = Image.fromarray(mask_array.astype(np.uint8))
            
            # Blend refined with original in center region
            refined_array = np.array(refined).astype(np.float32)
            original_array = np.array(original).astype(np.float32)
            mask_array = np.array(mask).astype(np.float32) / 255.0
            
            # Apply conservative blending (70% refined, 30% original in center)
            blended = (
                refined_array * (1 - mask_array * 0.3) + 
                original_array * (mask_array * 0.3)
            ).astype(np.uint8)
            
            return Image.fromarray(blended)
            
        except Exception as e:
            print(f"⚠️ Facial preservation error: {e}")
            return refined
    
    def _final_quality_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Apply final quality enhancement.
        
        Args:
            image: Input image
        
        Returns:
            Quality-enhanced image
        """
        # Subtle final enhancements
        enhanced = self.image_processor.adjust_contrast(image, 1.02)
        enhanced = self.image_processor.sharpen_image(enhanced, 1.05)
        
        return enhanced
    
    def compare_period_treatments(
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
            periods = list(self.PERIOD_REFINEMENTS.keys())
        
        comparison_results = {}
        
        for period in periods:
            try:
                refined, metadata = self.refine_historical_photo(image, period)
                comparison_results[period] = (refined, metadata)
            except Exception as e:
                print(f"⚠️ Error processing period {period}: {e}")
                comparison_results[period] = (image, {'error': str(e)})
        
        return comparison_results
    
    def get_available_periods(self) -> List[str]:
        """
        Get list of available historical periods.
        
        Returns:
            List of period names
        """
        return list(self.PERIOD_REFINEMENTS.keys())
    
    def get_period_info(self, period: str) -> Optional[Dict]:
        """
        Get detailed information about a specific period.
        
        Args:
            period: Period string
        
        Returns:
            Period information dictionary or None if invalid
        """
        return self.PERIOD_REFINEMENTS.get(period)


if __name__ == "__main__":
    # Test historical refinement
    print("Testing Historical Refinement...")
    
    try:
        refinement = HistoricalRefinement()
        
        # Test with image
        test_img = Image.new('RGB', (512, 512), color='blue')
        
        # Test historical refinement
        refined, metadata = refinement.refine_historical_photo(test_img, '1950s')
        print(f"✅ Historical refinement complete! Period: {metadata['period']}")
        print(f"✅ Applied steps: {[k for k in metadata.keys() if 'applied' in k]}")
        
        # Test period comparison
        comparison = refinement.compare_period_treatments(test_img, ['1920s', 'WWII', '1950s'])
        print(f"✅ Period comparison complete! Compared {len(comparison)} periods")
        
        # Test available periods
        periods = refinement.get_available_periods()
        print(f"✅ Available periods: {periods}")
        
        # Test period info
        period_info = refinement.get_period_info('1950s')
        print(f"✅ 1950s info: {period_info['name']}")
        
        print("✅ Historical refinement tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")