"""
Historical Era Detection Module for Task 4
Automatically detects historical era from image characteristics and provides manual override.
"""

import numpy as np
from PIL import Image, ImageStat
from typing import Dict, Tuple, Optional, List
import cv2


class EraDetector:
    """
    Detects the likely historical era of a photograph based on visual characteristics.
    Supports manual override and provides era-specific color palettes.
    """
    
    # Era definitions with time ranges and characteristics
    ERAS = {
        '1900s': {
            'range': (1900, 1909),
            'description': 'Early 1900s - Victorian/Edwardian era',
            'characteristics': ['sepia_tone', 'grainy', 'low_contrast', 'vignette'],
            'typical_colors': ['sepia', 'brown', 'warm_gray'],
            'saturation_range': (0.1, 0.3),
            'contrast_range': (0.8, 1.2),
            'grain_level': 'high'
        },
        '1910s': {
            'range': (1910, 1919),
            'description': '1910s - WWI era',
            'characteristics': ['sepia_tone', 'grainy', 'medium_contrast'],
            'typical_colors': ['sepia', 'brown', 'muted'],
            'saturation_range': (0.15, 0.35),
            'contrast_range': (0.9, 1.3),
            'grain_level': 'high'
        },
        '1920s': {
            'range': (1920, 1929),
            'description': '1920s - Roaring Twenties',
            'characteristics': ['soft_sepia', 'medium_grain', 'improving_quality'],
            'typical_colors': ['warm_sepia', 'soft_brown', 'muted_colors'],
            'saturation_range': (0.2, 0.4),
            'contrast_range': (1.0, 1.4),
            'grain_level': 'medium'
        },
        '1930s': {
            'range': (1930, 1939),
            'description': '1930s - Great Depression era',
            'characteristics': ['medium_sepia', 'medium_grain', 'depression_aesthetic'],
            'typical_colors': ['muted_sepia', 'gray_brown', 'desaturated'],
            'saturation_range': (0.15, 0.35),
            'contrast_range': (0.9, 1.2),
            'grain_level': 'medium'
        },
        '1940s': {
            'range': (1940, 1949),
            'description': '1940s - WWII era',
            'characteristics': ['medium_sepia', 'war_photography', 'mixed_quality'],
            'typical_colors': ['muted_sepia', 'military_tones', 'documentary_style'],
            'saturation_range': (0.2, 0.4),
            'contrast_range': (1.0, 1.3),
            'grain_level': 'medium'
        },
        '1950s': {
            'range': (1950, 1959),
            'description': '1950s - Post-war prosperity',
            'characteristics': ['mild_sepia', 'improving_quality', 'color_early_adoption'],
            'typical_colors': ['warm_tones', 'early_color', 'vibrant_beginnings'],
            'saturation_range': (0.3, 0.5),
            'contrast_range': (1.1, 1.4),
            'grain_level': 'low_medium'
        },
        '1960s': {
            'range': (1960, 1969),
            'description': '1960s - Cultural revolution',
            'characteristics': ['minimal_sepia', 'color_transition', 'vibrant_emerging'],
            'typical_colors': ['natural_colors', 'vibrant_accents', 'psychedelic_hints'],
            'saturation_range': (0.4, 0.6),
            'contrast_range': (1.1, 1.5),
            'grain_level': 'low'
        },
        '1970s': {
            'range': (1970, 1979),
            'description': '1970s - Color photography mainstream',
            'characteristics': ['natural_tones', 'warm_colors', 'film_characteristics'],
            'typical_colors': ['warm_colors', 'earth_tones', 'vintage_film'],
            'saturation_range': (0.5, 0.7),
            'contrast_range': (1.0, 1.3),
            'grain_level': 'low'
        }
    }
    
    def __init__(self):
        """Initialize the era detector."""
        print("📅 Initializing Historical Era Detector")
        print("✅ Era Detector Ready!")
    
    def detect_era(self, image: Image.Image) -> Dict:
        """
        Detect the likely historical era of an image based on visual characteristics.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            Dictionary with detected era and confidence scores
        """
        result = {
            'detected_era': '1950s',  # Default
            'confidence': 0.5,
            'all_scores': {},
            'image_characteristics': {},
            'manual_override_recommended': False
        }
        
        try:
            # Extract image characteristics
            characteristics = self._extract_image_characteristics(image)
            result['image_characteristics'] = characteristics
            
            # Score each era based on how well it matches the characteristics
            era_scores = {}
            for era_name, era_info in self.ERAS.items():
                score = self._score_era_match(characteristics, era_info)
                era_scores[era_name] = score
            
            result['all_scores'] = era_scores
            
            # Find the best matching era
            if era_scores:
                best_era = max(era_scores, key=era_scores.get)
                best_score = era_scores[best_era]
                
                result['detected_era'] = best_era
                result['confidence'] = best_score
                
                # Recommend manual override if confidence is low
                if best_score < 0.6:
                    result['manual_override_recommended'] = True
            
            return result
            
        except Exception as e:
            print(f"⚠️ Era detection error: {e}")
            result['error'] = str(e)
            return result
    
    def _extract_image_characteristics(self, image: Image.Image) -> Dict:
        """
        Extract visual characteristics from the image for era detection.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            Dictionary of image characteristics
        """
        characteristics = {}
        
        try:
            # Convert to grayscale for some analyses
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Calculate basic statistics
            stat = ImageStat.Stat(gray)
            characteristics['mean_brightness'] = stat.mean[0] / 255.0
            characteristics['stddev'] = stat.stddev[0] / 255.0
            
            # Estimate contrast
            characteristics['contrast'] = characteristics['stddev'] * 2.0
            
            # Estimate saturation (if color image)
            if image.mode == 'RGB':
                hsv = image.convert('HSV')
                h, s, v = hsv.split()
                sat_stat = ImageStat.Stat(s)
                characteristics['saturation'] = sat_stat.mean[0] / 255.0
            else:
                characteristics['saturation'] = 0.0
            
            # Estimate grain/noise level
            characteristics['grain_level'] = self._estimate_grain(gray)
            
            # Detect sepia tone
            characteristics['has_sepia'] = self._detect_sepia_tone(image)
            
            # Detect vignette
            characteristics['has_vignette'] = self._detect_vignette(gray)
            
            return characteristics
            
        except Exception as e:
            print(f"⚠️ Characteristic extraction error: {e}")
            return {}
    
    def _estimate_grain(self, gray_image: Image.Image) -> float:
        """
        Estimate the grain/noise level in a grayscale image.
        
        Args:
            gray_image: Grayscale PIL Image
        
        Returns:
            Grain level estimate (0.0-1.0)
        """
        try:
            # Convert to numpy array
            img_array = np.array(gray_image)
            
            # Apply Laplacian to detect edges/noise
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            grain_level = np.std(laplacian) / 255.0
            
            return min(1.0, grain_level * 2.0)  # Normalize to 0-1
            
        except Exception as e:
            return 0.5  # Default medium grain
    
    def _detect_sepia_tone(self, image: Image.Image) -> bool:
        """
        Detect if the image has a sepia tone.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            True if sepia tone detected
        """
        try:
            if image.mode != 'RGB':
                return False
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check for sepia characteristics (red > green > blue)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            # Sepia typically has R > G > B
            sepia_mask = (r > g) & (g > b)
            sepia_ratio = np.sum(sepia_mask) / (r.size)
            
            return sepia_ratio > 0.6
            
        except Exception as e:
            return False
    
    def _detect_vignette(self, gray_image: Image.Image) -> bool:
        """
        Detect if the image has vignette (darkened corners).
        
        Args:
            gray_image: Grayscale PIL Image
        
        Returns:
            True if vignette detected
        """
        try:
            # Convert to numpy array
            img_array = np.array(gray_image)
            
            # Calculate brightness at corners vs center
            height, width = img_array.shape
            corner_size = min(height, width) // 10
            
            # Sample corners
            corners = [
                img_array[:corner_size, :corner_size],  # Top-left
                img_array[:corner_size, -corner_size:],  # Top-right
                img_array[-corner_size:, :corner_size],  # Bottom-left
                img_array[-corner_size:, -corner_size:]  # Bottom-right
            ]
            
            # Sample center
            center = img_array[height//2-corner_size:height//2+corner_size, 
                             width//2-corner_size:width//2+corner_size]
            
            # Compare brightness
            corner_brightness = np.mean([np.mean(corner) for corner in corners])
            center_brightness = np.mean(center)
            
            # Vignette if corners are significantly darker than center
            return corner_brightness < center_brightness * 0.8
            
        except Exception as e:
            return False
    
    def _score_era_match(self, characteristics: Dict, era_info: Dict) -> float:
        """
        Score how well the image characteristics match an era.
        
        Args:
            characteristics: Extracted image characteristics
            era_info: Era information dictionary
        
        Returns:
            Match score (0.0-1.0)
        """
        score = 0.0
        total_factors = 0
        
        try:
            # Score saturation match
            if 'saturation' in characteristics and 'saturation_range' in era_info:
                sat_range = era_info['saturation_range']
                img_sat = characteristics['saturation']
                
                if sat_range[0] <= img_sat <= sat_range[1]:
                    score += 1.0
                elif sat_range[0] * 0.8 <= img_sat <= sat_range[1] * 1.2:
                    score += 0.5
                total_factors += 1
            
            # Score contrast match
            if 'contrast' in characteristics and 'contrast_range' in era_info:
                contrast_range = era_info['contrast_range']
                img_contrast = characteristics['contrast']
                
                if contrast_range[0] <= img_contrast <= contrast_range[1]:
                    score += 1.0
                elif contrast_range[0] * 0.8 <= img_contrast <= contrast_range[1] * 1.2:
                    score += 0.5
                total_factors += 1
            
            # Score grain level match
            if 'grain_level' in characteristics and 'grain_level' in era_info:
                era_grain = era_info['grain_level']
                img_grain = characteristics['grain_level']
                
                grain_mapping = {'low': 0.3, 'low_medium': 0.5, 'medium': 0.6, 'high': 0.8}
                expected_grain = grain_mapping.get(era_grain, 0.5)
                
                grain_diff = abs(img_grain - expected_grain)
                if grain_diff < 0.2:
                    score += 1.0
                elif grain_diff < 0.4:
                    score += 0.5
                total_factors += 1
            
            # Score sepia detection
            if 'has_sepia' in characteristics and 'sepia_tone' in era_info['characteristics']:
                if characteristics['has_sepia']:
                    score += 1.0
                total_factors += 1
            
            # Normalize score
            if total_factors > 0:
                score = score / total_factors
            
            return score
            
        except Exception as e:
            return 0.5  # Default neutral score
    
    def manual_override(self, era: str) -> Dict:
        """
        Set manual era override.
        
        Args:
            era: Era string (e.g., '1920s', '1950s')
        
        Returns:
            Dictionary with override information
        """
        if era not in self.ERAS:
            return {
                'success': False,
                'error': f'Invalid era: {era}. Available eras: {list(self.ERAS.keys())}'
            }
        
        return {
            'success': True,
            'era': era,
            'era_info': self.ERAS[era],
            'description': self.ERAS[era]['description']
        }
    
    def get_available_eras(self) -> List[str]:
        """
        Get list of available historical eras.
        
        Returns:
            List of era names
        """
        return list(self.ERAS.keys())
    
    def get_era_info(self, era: str) -> Optional[Dict]:
        """
        Get detailed information about a specific era.
        
        Args:
            era: Era string
        
        Returns:
            Era information dictionary or None if invalid
        """
        return self.ERAS.get(era)


if __name__ == "__main__":
    # Test era detector
    print("Testing Era Detector...")
    
    try:
        detector = EraDetector()
        
        # Test with a grayscale image
        test_img = Image.new('RGB', (512, 512), color='gray')
        
        # Test era detection
        result = detector.detect_era(test_img)
        print(f"✅ Detected era: {result['detected_era']}")
        print(f"✅ Confidence: {result['confidence']:.2f}")
        print(f"✅ Image characteristics: {result['image_characteristics']}")
        
        # Test manual override
        override = detector.manual_override('1920s')
        print(f"✅ Manual override: {override}")
        
        # Test available eras
        eras = detector.get_available_eras()
        print(f"✅ Available eras: {eras}")
        
        # Test era info
        era_info = detector.get_era_info('1950s')
        print(f"✅ 1950s info: {era_info['description']}")
        
        print("✅ Era detector tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")