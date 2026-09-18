"""
Cross-Domain Image Colorization Pipeline for Task 5
Supports multiple input domains with domain-specific preprocessing and colorization.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Tuple, Optional, List
import cv2

from pipelines.base_colorizer import BaseColorizer
from processing.image_processor import ImageProcessor


class CrossDomainColorizer:
    """
    Cross-domain colorization system that adapts to different input domains.
    Supports grayscale photographs, sketches, infrared/satellite images, etc.
    """
    
    # Supported domains and their characteristics
    DOMAINS = {
        'grayscale_photo': {
            'name': 'Grayscale Photograph',
            'description': 'Standard black and white photographs',
            'characteristics': ['continuous_tones', 'natural_lighting', 'photographic_quality'],
            'preprocessing': 'standard',
            'colorization_approach': 'photorealistic',
            'recommended_strength': 0.75
        },
        'sketch': {
            'name': 'Black and White Sketch',
            'description': 'Line drawings and sketches',
            'characteristics': ['line_art', 'high_contrast_edges', 'minimal_shading'],
            'preprocessing': 'sketch_enhancement',
            'colorization_approach': 'artistic',
            'recommended_strength': 0.85
        },
        'infrared': {
            'name': 'Infrared/Satellite Image',
            'description': 'Infrared photography and satellite imagery',
            'characteristics': ['thermal_contrast', 'unnatural_brightness', 'scientific_imaging'],
            'preprocessing': 'infrared_normalization',
            'colorization_approach': 'false_color',
            'recommended_strength': 0.70
        },
        'blueprint': {
            'name': 'Blueprint/Technical Drawing',
            'description': 'Technical drawings and blueprints',
            'characteristics': ['white_lines', 'blue_background', 'technical_precision'],
            'preprocessing': 'blueprint_conversion',
            'colorization_approach': 'technical',
            'recommended_strength': 0.80
        },
        'newspaper': {
            'name': 'Newspaper Print',
            'description': 'Halftone newspaper prints',
            'characteristics': ['halftone_pattern', 'low_quality', 'high_contrast'],
            'preprocessing': 'newspaper_cleaning',
            'colorization_approach': 'documentary',
            'recommended_strength': 0.65
        },
        'xray': {
            'name': 'X-Ray/Medical Image',
            'description': 'Medical X-ray images',
            'characteristics': ['inverted_grayscale', 'high_contrast', 'medical_imaging'],
            'preprocessing': 'xray_normalization',
            'colorization_approach': 'medical',
            'recommended_strength': 0.60
        }
    }
    
    def __init__(self, device: str = "auto"):
        """
        Initialize cross-domain colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"🌐 Initializing Cross-Domain Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            print("✅ Cross-Domain Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize cross-domain colorizer: {e}")
            raise
    
    def detect_domain(self, image: Image.Image) -> Dict:
        """
        Automatically detect the likely domain of an input image.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Dictionary with detected domain and confidence
        """
        result = {
            'detected_domain': 'grayscale_photo',  # Default
            'confidence': 0.5,
            'all_scores': {},
            'image_characteristics': {}
        }
        
        try:
            # Extract image characteristics
            characteristics = self._extract_domain_characteristics(image)
            result['image_characteristics'] = characteristics
            
            # Score each domain based on characteristics
            domain_scores = {}
            for domain_name, domain_info in self.DOMAINS.items():
                score = self._score_domain_match(characteristics, domain_info)
                domain_scores[domain_name] = score
            
            result['all_scores'] = domain_scores
            
            # Find the best matching domain
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                best_score = domain_scores[best_domain]
                
                result['detected_domain'] = best_domain
                result['confidence'] = best_score
            
            return result
            
        except Exception as e:
            print(f"⚠️ Domain detection error: {e}")
            result['error'] = str(e)
            return result
    
    def _extract_domain_characteristics(self, image: Image.Image) -> Dict:
        """
        Extract characteristics for domain detection.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Dictionary of image characteristics
        """
        characteristics = {}
        
        try:
            # Convert to grayscale for analysis
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Calculate basic statistics
            img_array = np.array(gray)
            characteristics['mean_brightness'] = np.mean(img_array) / 255.0
            characteristics['stddev'] = np.std(img_array) / 255.0
            characteristics['contrast'] = characteristics['stddev'] * 2.0
            
            # Edge detection
            edges = cv2.Canny(img_array, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            characteristics['edge_density'] = edge_ratio
            
            # Line detection (for sketches/blueprints)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            characteristics['line_count'] = line_count
            
            # Check for halftone pattern (newspaper)
            # Look for regular dot patterns using frequency analysis
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            characteristics['halftone_indicator'] = np.std(magnitude_spectrum)
            
            # Check for inverted grayscale (X-ray)
            characteristics['is_inverted'] = characteristics['mean_brightness'] > 0.7
            
            # Check for blue background (blueprint)
            if image.mode == 'RGB':
                r, g, b = image.split()
                b_stat = Image.Image.stat(b)
                characteristics['blue_dominance'] = b_stat.mean[0] / 255.0
            
            return characteristics
            
        except Exception as e:
            print(f"⚠️ Characteristic extraction error: {e}")
            return {}
    
    def _score_domain_match(self, characteristics: Dict, domain_info: Dict) -> float:
        """
        Score how well image characteristics match a domain.
        
        Args:
            characteristics: Extracted image characteristics
            domain_info: Domain information dictionary
        
        Returns:
            Match score (0.0-1.0)
        """
        score = 0.0
        total_factors = 0
        
        try:
            domain_chars = domain_info.get('characteristics', [])
            
            # Edge density scoring
            if 'edge_density' in characteristics:
                edge_density = characteristics['edge_density']
                
                if 'line_art' in domain_chars or 'high_contrast_edges' in domain_chars:
                    # Sketches and blueprints have high edge density
                    if edge_density > 0.15:
                        score += 1.0
                    elif edge_density > 0.10:
                        score += 0.5
                    total_factors += 1
                elif 'continuous_tones' in domain_chars:
                    # Photos have lower edge density
                    if edge_density < 0.10:
                        score += 1.0
                    elif edge_density < 0.15:
                        score += 0.5
                    total_factors += 1
            
            # Line count scoring
            if 'line_count' in characteristics:
                line_count = characteristics['line_count']
                
                if 'technical_precision' in domain_chars:
                    # Blueprints have many straight lines
                    if line_count > 20:
                        score += 1.0
                    elif line_count > 10:
                        score += 0.5
                    total_factors += 1
            
            # Halftone pattern scoring
            if 'halftone_indicator' in characteristics:
                halftone = characteristics['halftone_indicator']
                
                if 'halftone_pattern' in domain_chars:
                    # Newspaper prints have strong halftone patterns
                    if halftone > 2.0:
                        score += 1.0
                    elif halftone > 1.5:
                        score += 0.5
                    total_factors += 1
            
            # Inverted grayscale scoring
            if 'is_inverted' in characteristics:
                is_inverted = characteristics['is_inverted']
                
                if 'inverted_grayscale' in domain_chars:
                    if is_inverted:
                        score += 1.0
                    total_factors += 1
            
            # Blue dominance scoring
            if 'blue_dominance' in characteristics:
                blue_dominance = characteristics['blue_dominance']
                
                if 'blue_background' in domain_chars:
                    if blue_dominance > 0.6:
                        score += 1.0
                    elif blue_dominance > 0.4:
                        score += 0.5
                    total_factors += 1
            
            # Normalize score
            if total_factors > 0:
                score = score / total_factors
            
            return score
            
        except Exception as e:
            return 0.5  # Default neutral score
    
    def preprocess_domain(self, image: Image.Image, domain: str) -> Tuple[Image.Image, Dict]:
        """
        Apply domain-specific preprocessing to the image.
        
        Args:
            image: Input PIL Image
            domain: Domain string
        
        Returns:
            Tuple of (preprocessed_image, metadata_dict)
        """
        metadata = {
            'domain': domain,
            'preprocessing_method': self.DOMAINS[domain]['preprocessing']
        }
        
        try:
            preprocessing_method = self.DOMAINS[domain]['preprocessing']
            
            if preprocessing_method == 'standard':
                # Standard photographic preprocessing
                preprocessed = self._preprocess_standard(image)
            
            elif preprocessing_method == 'sketch_enhancement':
                # Sketch-specific preprocessing
                preprocessed = self.image_processor.preprocess_sketch(image)
            
            elif preprocessing_method == 'infrared_normalization':
                # Infrared/satellite preprocessing
                preprocessed = self.image_processor.preprocess_infrared(image)
            
            elif preprocessing_method == 'blueprint_conversion':
                # Blueprint preprocessing
                preprocessed = self._preprocess_blueprint(image)
            
            elif preprocessing_method == 'newspaper_cleaning':
                # Newspaper preprocessing
                preprocessed = self._preprocess_newspaper(image)
            
            elif preprocessing_method == 'xray_normalization':
                # X-ray preprocessing
                preprocessed = self._preprocess_xray(image)
            
            else:
                # Default to standard
                preprocessed = self._preprocess_standard(image)
            
            return preprocessed, metadata
            
        except Exception as e:
            print(f"⚠️ Domain preprocessing error: {e}")
            metadata['error'] = str(e)
            return image, metadata
    
    def _preprocess_standard(self, image: Image.Image) -> Image.Image:
        """Standard photographic preprocessing."""
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Light denoising
        preprocessed = self.image_processor.denoise_image(image, strength=1)
        
        # Slight contrast enhancement
        preprocessed = self.image_processor.adjust_contrast(preprocessed, 1.05)
        
        return preprocessed
    
    def _preprocess_blueprint(self, image: Image.Image) -> Image.Image:
        """Blueprint-specific preprocessing."""
        # Convert to grayscale and invert
        if image.mode == 'RGB':
            gray = image.convert('L')
        else:
            gray = image
        
        # Invert to get white lines on dark background
        inverted = Image.eval(gray, lambda x: 255 - x)
        
        # Enhance edges
        enhanced = self.image_processor.sharpen_image(inverted, 2.0)
        
        # Convert back to RGB
        preprocessed = enhanced.convert('RGB')
        
        return preprocessed
    
    def _preprocess_newspaper(self, image: Image.Image) -> Image.Image:
        """Newspaper-specific preprocessing."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Apply blur to reduce halftone pattern
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Enhance contrast
        enhanced = self.image_processor.adjust_contrast(blurred, 1.2)
        
        # Convert to RGB
        preprocessed = enhanced.convert('RGB')
        
        return preprocessed
    
    def _preprocess_xray(self, image: Image.Image) -> Image.Image:
        """X-ray-specific preprocessing."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Invert if needed (X-rays are typically inverted)
        img_array = np.array(gray)
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
        
        # Normalize contrast
        normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to PIL
        preprocessed = Image.fromarray(normalized).convert('RGB')
        
        return preprocessed
    
    def colorize_cross_domain(
        self,
        image: Image.Image,
        domain: Optional[str] = None,
        auto_detect: bool = True,
        strength: Optional[float] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        Colorize an image with domain-specific processing.
        
        Args:
            image: Input PIL Image
            domain: Specific domain (None = auto-detect)
            auto_detect: Whether to auto-detect domain if not specified
            strength: Colorization strength (None = use domain recommendation)
        
        Returns:
            Tuple of (colorized_image, metadata_dict)
        """
        metadata = {
            'method': 'cross_domain_colorization',
            'auto_detect': auto_detect
        }
        
        try:
            # Determine domain
            if domain:
                if domain not in self.DOMAINS:
                    raise ValueError(f"Invalid domain: {domain}. Available: {list(self.DOMAINS.keys())}")
                selected_domain = domain
                metadata['domain_selection'] = 'manual'
            elif auto_detect:
                detection_result = self.detect_domain(image)
                selected_domain = detection_result['detected_domain']
                metadata['domain_selection'] = 'auto_detected'
                metadata['domain_detection'] = detection_result
            else:
                selected_domain = 'grayscale_photo'  # Default
                metadata['domain_selection'] = 'default'
            
            metadata['selected_domain'] = selected_domain
            
            # Get domain information
            domain_info = self.DOMAINS[selected_domain]
            metadata['domain_info'] = domain_info
            
            # Apply domain-specific preprocessing
            preprocessed, preprocess_metadata = self.preprocess_domain(image, selected_domain)
            metadata['preprocessing'] = preprocess_metadata
            
            # Determine colorization strength
            if strength is None:
                strength = domain_info['recommended_strength']
            metadata['strength'] = strength
            
            # Create domain-specific colorization prompt
            domain_prompt = self._create_domain_specific_prompt(selected_domain)
            
            # Apply colorization
            colorized, color_metadata = self.colorizer.colorize(
                preprocessed,
                prompt=domain_prompt,
                strength=strength
            )
            
            metadata['colorization_metadata'] = color_metadata
            metadata['domain_prompt_used'] = domain_prompt
            
            # Apply domain-specific postprocessing
            final = self._postprocess_domain(colorized, selected_domain)
            metadata['postprocessing'] = True
            
            return final, metadata
            
        except Exception as e:
            print(f"⚠️ Cross-domain colorization error: {e}")
            metadata['error'] = str(e)
            # Fallback to basic colorization
            colorized, color_metadata = self.colorizer.colorize(image)
            metadata['fallback'] = 'basic_colorization'
            return colorized, metadata
    
    def _create_domain_specific_prompt(self, domain: str) -> str:
        """
        Create a colorization prompt specific to the domain.
        
        Args:
            domain: Domain string
        
        Returns:
            Domain-specific colorization prompt
        """
        base_prompt = "colorize this image, realistic colors, high quality"
        
        domain_prompts = {
            'grayscale_photo': "photorealistic colorization, natural colors, photographic quality",
            'sketch': "artistic colorization, creative colors, respect line art structure",
            'infrared': "false color representation, scientific visualization, thermal color mapping",
            'blueprint': "technical colorization, precise colors, maintain technical clarity",
            'newspaper': "documentary colorization, historical accuracy, vintage newspaper aesthetic",
            'xray': "medical colorization, anatomical accuracy, medical imaging colors"
        }
        
        domain_instruction = domain_prompts.get(domain, "realistic colorization")
        
        return f"{base_prompt}, {domain_instruction}"
    
    def _postprocess_domain(self, image: Image.Image, domain: str) -> Image.Image:
        """
        Apply domain-specific postprocessing.
        
        Args:
            image: Colorized image
            domain: Domain string
        
        Returns:
            Postprocessed image
        """
        if domain == 'sketch':
            # Maintain line art clarity
            return self.image_processor.sharpen_image(image, 1.3)
        elif domain == 'infrared':
            # Enhance color contrast
            return self.image_processor.adjust_saturation(image, 1.2)
        elif domain == 'newspaper':
            # Add subtle vintage effect
            return self.image_processor.sepia_tone(image, 0.1)
        else:
            return image
    
    def get_available_domains(self) -> List[str]:
        """
        Get list of available domains.
        
        Returns:
            List of domain names
        """
        return list(self.DOMAINS.keys())
    
    def get_domain_info(self, domain: str) -> Optional[Dict]:
        """
        Get detailed information about a specific domain.
        
        Args:
            domain: Domain string
        
        Returns:
            Domain information dictionary or None if invalid
        """
        return self.DOMAINS.get(domain)


if __name__ == "__main__":
    # Test cross-domain colorizer
    print("Testing Cross-Domain Colorizer...")
    
    try:
        colorizer = CrossDomainColorizer()
        
        # Test with different image types
        test_photo = Image.new('RGB', (512, 512), color='gray')
        test_sketch = Image.new('RGB', (512, 512), color='white')
        
        # Test domain detection
        domain_result = colorizer.detect_domain(test_photo)
        print(f"✅ Detected domain: {domain_result['detected_domain']}")
        print(f"✅ Confidence: {domain_result['confidence']:.2f}")
        
        # Test cross-domain colorization with auto-detection
        colorized, metadata = colorizer.colorize_cross_domain(test_photo)
        print(f"✅ Cross-domain colorization complete! Domain: {metadata['selected_domain']}")
        
        # Test with manual domain selection
        colorized_sketch, metadata_sketch = colorizer.colorize_cross_domain(
            test_sketch, domain='sketch', auto_detect=False
        )
        print(f"✅ Sketch colorization complete! Domain: {metadata_sketch['selected_domain']}")
        
        # Test available domains
        domains = colorizer.get_available_domains()
        print(f"✅ Available domains: {domains}")
        
        # Test domain info
        sketch_info = colorizer.get_domain_info('sketch')
        print(f"✅ Sketch domain: {sketch_info['name']}")
        
        print("✅ Cross-domain colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")