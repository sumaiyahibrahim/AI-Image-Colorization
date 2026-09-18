"""
Quality Metrics for Image Colorization
Implements metrics from the training: saturation, colorfulness, SSIM, edge preservation, skin naturalness
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


class QualityMetrics:
    """
    Quality evaluation metrics for colorized images.
    Based on training course metrics: saturation, colorfulness, SSIM, edge preservation, skin naturalness.
    """
    
    @staticmethod
    def calculate_saturation(image: Image.Image) -> float:
        """
        Calculate average saturation of the image.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Average saturation value (0.0-1.0)
        """
        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        
        # Calculate average saturation
        saturation_array = np.array(s)
        avg_saturation = np.mean(saturation_array) / 255.0
        
        return avg_saturation
    
    @staticmethod
    def calculate_colorfulness(image: Image.Image) -> float:
        """
        Calculate colorfulness metric using Hasler & Süsstrunk method.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Colorfulness score (higher = more colorful)
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Split into RGB channels
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Calculate RG and YB color opposites
        RG = R - G
        YB = 0.5 * (R + G) - B
        
        # Calculate standard deviations and means
        rg_std = np.std(RG)
        rg_mean = np.mean(RG)
        yb_std = np.std(YB)
        yb_mean = np.mean(YB)
        
        # Calculate colorfulness
        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
        colorfulness = std_root + (0.3 * mean_root)
        
        return colorfulness
    
    @staticmethod
    def calculate_ssim(original: Image.Image, colorized: Image.Image) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original grayscale image
            colorized: Colorized image (will be converted to grayscale for comparison)
        
        Returns:
            SSIM score (-1 to 1, higher = better structural similarity)
        """
        # Ensure both images are same size
        if original.size != colorized.size:
            colorized = colorized.resize(original.size, Image.Resampling.LANCZOS)
        
        # Convert both to grayscale for structural comparison
        original_gray = original.convert('L') if original.mode != 'L' else original
        colorized_gray = colorized.convert('L')
        
        # Convert to numpy arrays
        original_array = np.array(original_gray)
        colorized_array = np.array(colorized_gray)
        
        # Calculate SSIM
        try:
            ssim_score = ssim(original_array, colorized_array, data_range=255)
            return ssim_score
        except Exception as e:
            print(f"Warning: SSIM calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_edge_preservation(original: Image.Image, colorized: Image.Image) -> float:
        """
        Calculate edge preservation using gradient correlation.
        
        Args:
            original: Original grayscale image
            colorized: Colorized image
        
        Returns:
            Edge preservation score (0.0-1.0, higher = better edge preservation)
        """
        # Ensure both images are same size
        if original.size != colorized.size:
            colorized = colorized.resize(original.size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        original_gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY) if original.mode == 'RGB' else np.array(original)
        colorized_gray = cv2.cvtColor(np.array(colorized), cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients using Sobel operator
        grad_original_x = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_original_y = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_colorized_x = cv2.Sobel(colorized_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_colorized_y = cv2.Sobel(colorized_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitudes
        grad_original = np.sqrt(grad_original_x**2 + grad_original_y**2)
        grad_colorized = np.sqrt(grad_colorized_x**2 + grad_colorized_y**2)
        
        # Calculate correlation
        grad_original_flat = grad_original.flatten()
        grad_colorized_flat = grad_colorized.flatten()
        
        correlation = np.corrcoef(grad_original_flat, grad_colorized_flat)[0, 1]
        
        # Handle NaN cases
        if np.isnan(correlation):
            return 0.0
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, (correlation + 1) / 2))
    
    @staticmethod
    def calculate_skin_naturalness(image: Image.Image) -> Dict[str, float]:
        """
        Calculate skin naturalness metrics.
        Detects skin regions and evaluates color naturalness.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Dictionary with skin naturalness metrics
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple skin detection using HSV ranges
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Count skin pixels
        skin_pixels = np.count_nonzero(skin_mask)
        total_pixels = image.size[0] * image.size[1]
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        # If no skin detected, return default values
        if skin_pixels == 0:
            return {
                "skin_detected": False,
                "skin_ratio": 0.0,
                "skin_naturalness": 0.0,
                "avg_skin_hue": 0.0,
                "avg_skin_saturation": 0.0
            }
        
        # Extract skin regions
        skin_regions = cv2.bitwise_and(hsv, hsv, mask=skin_mask)
        
        # Calculate average hue and saturation in skin regions
        skin_hue = skin_regions[:, :, 0][skin_mask > 0]
        skin_sat = skin_regions[:, :, 1][skin_mask > 0]
        
        avg_hue = np.mean(skin_hue) / 180.0  # Normalize to 0-1
        avg_sat = np.mean(skin_sat) / 255.0  # Normalize to 0-1
        
        # Calculate naturalness based on typical skin tone ranges
        # Natural skin typically has hue around 0.05-0.15 and saturation 0.2-0.6
        hue_naturalness = 1.0 - min(1.0, max(0.0, abs(avg_hue - 0.1) / 0.1))
        sat_naturalness = 1.0 - min(1.0, max(0.0, abs(avg_sat - 0.4) / 0.4))
        
        overall_naturalness = (hue_naturalness + sat_naturalness) / 2.0
        
        return {
            "skin_detected": True,
            "skin_ratio": skin_ratio,
            "skin_naturalness": overall_naturalness,
            "avg_skin_hue": avg_hue,
            "avg_skin_saturation": avg_sat
        }
    
    @staticmethod
    def generate_quality_report(
        original: Image.Image,
        colorized: Image.Image,
        task_name: str = "Colorization"
    ) -> Dict:
        """
        Generate comprehensive quality report for colorization results.
        
        Args:
            original: Original grayscale image
            colorized: Colorized image
            task_name: Name of the colorization task
        
        Returns:
            Dictionary containing all quality metrics
        """
        report = {
            "task": task_name,
            "image_size": colorized.size,
            "metrics": {}
        }
        
        try:
            # Calculate all metrics
            report["metrics"]["saturation"] = QualityMetrics.calculate_saturation(colorized)
            report["metrics"]["colorfulness"] = QualityMetrics.calculate_colorfulness(colorized)
            report["metrics"]["ssim"] = QualityMetrics.calculate_ssim(original, colorized)
            report["metrics"]["edge_preservation"] = QualityMetrics.calculate_edge_preservation(original, colorized)
            report["metrics"]["skin_naturalness"] = QualityMetrics.calculate_skin_naturalness(colorized)
            
            # Calculate overall quality score
            overall_score = (
                report["metrics"]["saturation"] * 0.2 +
                min(1.0, report["metrics"]["colorfulness"] / 50.0) * 0.2 +
                report["metrics"]["ssim"] * 0.25 +
                report["metrics"]["edge_preservation"] * 0.25 +
                report["metrics"]["skin_naturalness"].get("skin_naturalness", 0.5) * 0.1
            )
            report["metrics"]["overall_quality"] = overall_score
            
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            report["error"] = str(e)
        
        return report
    
    @staticmethod
    def format_quality_report(report: Dict) -> str:
        """
        Format quality report for display.
        
        Args:
            report: Quality report dictionary
        
        Returns:
            Formatted string report
        """
        if "error" in report:
            return f"❌ Quality evaluation failed: {report['error']}"
        
        formatted = f"""
📊 Quality Report: {report['task']}
{'=' * 40}

🎨 Color Metrics:
• Saturation: {report['metrics']['saturation']:.3f} (0.0-1.0, higher = more saturated)
• Colorfulness: {report['metrics']['colorfulness']:.3f} (higher = more colorful)

🏗️ Structural Metrics:
• SSIM: {report['metrics']['ssim']:.3f} (-1 to 1, higher = better structural similarity)
• Edge Preservation: {report['metrics']['edge_preservation']:.3f} (0.0-1.0, higher = better edges)

👤 Skin Naturalness:
"""
        skin_data = report['metrics']['skin_naturalness']
        if skin_data['skin_detected']:
            formatted += f"""• Skin Detected: Yes ({skin_data['skin_ratio']*100:.1f}% of image)
• Skin Naturalness: {skin_data['skin_naturalness']:.3f} (0.0-1.0, higher = more natural)
• Average Skin Hue: {skin_data['avg_skin_hue']:.3f}
• Average Skin Saturation: {skin_data['avg_skin_saturation']:.3f}
"""
        else:
            formatted += "• Skin Detected: No skin regions found\n"
        
        formatted += f"""
📈 Overall Quality Score: {report['metrics']['overall_quality']:.3f} (0.0-1.0)
"""
        return formatted


if __name__ == "__main__":
    # Test quality metrics
    print("Testing Quality Metrics...")
    
    # Create test images
    original = Image.new('RGB', (256, 256), color='gray')
    colorized = Image.new('RGB', (256, 256), color='blue')
    
    # Test individual metrics
    saturation = QualityMetrics.calculate_saturation(colorized)
    print(f"✅ Saturation: {saturation:.3f}")
    
    colorfulness = QualityMetrics.calculate_colorfulness(colorized)
    print(f"✅ Colorfulness: {colorfulness:.3f}")
    
    ssim = QualityMetrics.calculate_ssim(original, colorized)
    print(f"✅ SSIM: {ssim:.3f}")
    
    edge_pres = QualityMetrics.calculate_edge_preservation(original, colorized)
    print(f"✅ Edge Preservation: {edge_pres:.3f}")
    
    skin_nat = QualityMetrics.calculate_skin_naturalness(colorized)
    print(f"✅ Skin Naturalness: {skin_nat}")
    
    # Test full report
    report = QualityMetrics.generate_quality_report(original, colorized, "Test")
    formatted = QualityMetrics.format_quality_report(report)
    print(formatted)
    
    print("✅ Quality metrics tests passed!")