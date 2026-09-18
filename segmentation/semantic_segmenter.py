"""
Semantic Segmentation Module for Multi-Object Colorization
Uses pretrained DeepLabV3 model for object detection and segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms


class SemanticSegmenter:
    """
    Semantic segmentation using pretrained DeepLabV3 model.
    Supports COCO dataset classes for multi-object colorization.
    """
    
    # COCO dataset class names and color mappings
    COCO_CLASSES = {
        0: 'background',
        1: 'person', 
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        26: 'backpack',
        27: 'umbrella',
        28: 'handbag',
        29: 'tie',
        30: 'suitcase',
        31: 'frisbee',
        32: 'skis',
        33: 'snowboard',
        34: 'sports ball',
        35: 'kite',
        36: 'baseball bat',
        37: 'baseball glove',
        38: 'skateboard',
        39: 'surfboard',
        40: 'tennis racket',
        41: 'bottle',
        42: 'wine glass',
        43: 'cup',
        44: 'fork',
        45: 'knife',
        46: 'spoon',
        47: 'bowl',
        48: 'banana',
        49: 'apple',
        50: 'sandwich',
        51: 'orange',
        52: 'broccoli',
        53: 'carrot',
        54: 'hot dog',
        55: 'pizza',
        56: 'donut',
        57: 'cake',
        58: 'chair',
        59: 'couch',
        60: 'potted plant',
        61: 'bed',
        62: 'dining table',
        63: 'toilet',
        64: 'tv',
        65: 'laptop',
        66: 'mouse',
        67: 'remote',
        68: 'keyboard',
        69: 'cell phone',
        70: 'microwave',
        71: 'oven',
        72: 'toaster',
        73: 'sink',
        74: 'refrigerator',
        75: 'book',
        76: 'clock',
        77: 'vase',
        78: 'scissors',
        79: 'teddy bear',
        80: 'hair drier',
        81: 'toothbrush'
    }
    
    # Simplified color mapping for main categories
    CATEGORY_COLORS = {
        'person': (255, 200, 150),        # Skin tone
        'vehicle': (100, 150, 255),       # Blue for vehicles
        'animal': (200, 100, 100),       # Red-ish for animals
        'nature': (100, 200, 100),       # Green for nature
        'building': (150, 150, 150),      # Gray for buildings
        'sky': (135, 206, 235),          # Sky blue
        'ground': (139, 69, 19),         # Brown for ground
        'object': (200, 200, 100)        # Yellow for other objects
    }
    
    # Mapping from COCO classes to simplified categories
    CLASS_TO_CATEGORY = {
        'person': 'person',
        'bicycle': 'vehicle',
        'car': 'vehicle', 
        'motorcycle': 'vehicle',
        'airplane': 'vehicle',
        'bus': 'vehicle',
        'train': 'vehicle',
        'truck': 'vehicle',
        'boat': 'vehicle',
        'bird': 'animal',
        'cat': 'animal',
        'dog': 'animal',
        'horse': 'animal',
        'sheep': 'animal',
        'cow': 'animal',
        'elephant': 'animal',
        'bear': 'animal',
        'zebra': 'animal',
        'giraffe': 'animal',
        'potted plant': 'nature',
        'bench': 'building',
        'chair': 'building',
        'couch': 'building',
        'bed': 'building',
        'dining table': 'building',
        'toilet': 'building',
        'tv': 'building',
        'laptop': 'object',
        'cell phone': 'object',
        'book': 'object',
        'clock': 'object',
        'vase': 'object'
    }
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the semantic segmentation model.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        
        print(f"🔍 Initializing Semantic Segmenter on {self.device}")
        
        try:
            # Load pretrained DeepLabV3 model
            from torchvision.models.segmentation import deeplabv3_resnet50
            
            self.model = deeplabv3_resnet50(pretrained=True)
            self.model.eval()
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                print("  ✓ Model loaded on GPU")
            else:
                print("  ✓ Model loaded on CPU")
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ Semantic Segmenter Ready!")
            
        except Exception as e:
            print(f"❌ Failed to load segmentation model: {e}")
            raise
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 GPU Detected: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("💻 Using CPU (GPU not available)")
        return torch.device(device)
    
    def segment_image(self, image: Image.Image) -> Tuple[np.ndarray, Dict]:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: PIL Image to segment
        
        Returns:
            Tuple of (segmentation_mask, detected_objects_dict)
        """
        # Preprocess image
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0)
        
        if self.device.type == "cuda":
            image_tensor = image_tensor.to(self.device)
        
        # Perform segmentation
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Get segmentation mask
        if isinstance(output, dict):
            output = output['out']
        
        # Get predicted class for each pixel
        predictions = output.argmax(1).squeeze(0).cpu().numpy()
        
        # Resize to original image size
        mask = cv2.resize(predictions, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Detect objects and their properties
        detected_objects = self._analyze_segments(mask)
        
        return mask, detected_objects
    
    def _analyze_segments(self, mask: np.ndarray) -> Dict:
        """
        Analyze segmentation mask to extract object information.
        
        Args:
            mask: Segmentation mask with class indices
        
        Returns:
            Dictionary with detected objects and their properties
        """
        detected_objects = {
            'objects': [],
            'class_counts': {},
            'dominant_classes': []
        }
        
        # Count pixels per class
        unique_classes, counts = np.unique(mask, return_counts=True)
        
        # Sort by pixel count (most dominant first)
        sorted_indices = np.argsort(counts)[::-1]
        
        for idx in sorted_indices[:10]:  # Top 10 classes
            class_id = unique_classes[idx]
            pixel_count = counts[idx]
            
            if class_id in self.COCO_CLASSES:
                class_name = self.COCO_CLASSES[class_id]
                
                detected_objects['class_counts'][class_name] = pixel_count
                detected_objects['dominant_classes'].append({
                    'class': class_name,
                    'pixels': pixel_count,
                    'percentage': (pixel_count / mask.size) * 100
                })
                
                # Get category and color
                category = self.CLASS_TO_CATEGORY.get(class_name, 'object')
                color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS['object'])
                
                detected_objects['objects'].append({
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'category': category,
                    'color': color,
                    'pixel_count': pixel_count,
                    'percentage': (pixel_count / mask.size) * 100
                })
        
        return detected_objects
    
    def get_object_masks(self, mask: np.ndarray, class_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Extract binary masks for specific object classes.
        
        Args:
            mask: Full segmentation mask
            class_ids: List of class IDs to extract
        
        Returns:
            Dictionary mapping class IDs to binary masks
        """
        object_masks = {}
        
        for class_id in class_ids:
            binary_mask = (mask == class_id).astype(np.uint8)
            object_masks[class_id] = binary_mask
        
        return object_masks
    
    def apply_object_colors(self, image: Image.Image, mask: np.ndarray, 
                           detected_objects: Dict) -> Image.Image:
        """
        Apply category-specific colors to objects in the image.
        
        Args:
            image: Original PIL Image
            mask: Segmentation mask
            detected_objects: Object information from segment_image
        
        Returns:
            PIL Image with object-specific colors applied
        """
        # Convert image to numpy array
        img_array = np.array(image).copy()
        
        # Apply colors for each detected object
        for obj in detected_objects['objects']:
            class_id = obj['class_id']
            color = obj['color']
            
            # Create mask for this object
            object_mask = (mask == class_id)
            
            # Apply color to object regions
            if np.any(object_mask):
                img_array[object_mask] = color
        
        return Image.fromarray(img_array)
    
    def create_segmentation_overlay(self, image: Image.Image, mask: np.ndarray, 
                                   alpha: float = 0.5) -> Image.Image:
        """
        Create a visualization overlay of segmentation on the original image.
        
        Args:
            image: Original PIL Image
            mask: Segmentation mask
            alpha: Transparency of overlay (0.0-1.0)
        
        Returns:
            PIL Image with segmentation overlay
        """
        # Create colored segmentation map
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id, class_name in self.COCO_CLASSES.items():
            category = self.CLASS_TO_CATEGORY.get(class_name, 'object')
            color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS['object'])
            colored_mask[mask == class_id] = color
        
        # Blend with original image
        original_array = np.array(image)
        overlay_array = cv2.addWeighted(original_array, 1-alpha, colored_mask, alpha, 0)
        
        return Image.fromarray(overlay_array)
    
    def get_colorization_prompts(self, detected_objects: Dict) -> List[str]:
        """
        Generate colorization prompts based on detected objects.
        
        Args:
            detected_objects: Object information from segment_image
        
        Returns:
            List of colorization prompts for different objects
        """
        prompts = []
        
        for obj in detected_objects['objects']:
            class_name = obj['class_name']
            category = obj['category']
            color = obj['color']
            
            # Generate prompt based on category
            if category == 'person':
                prompts.append(f"realistic skin tones for {class_name}")
            elif category == 'vehicle':
                prompts.append(f"realistic colors for {class_name}")
            elif category == 'nature':
                prompts.append(f"natural green colors for {class_name}")
            elif category == 'building':
                prompts.append(f"realistic building colors for {class_name}")
            else:
                prompts.append(f"appropriate colors for {class_name}")
        
        return prompts


if __name__ == "__main__":
    # Test the semantic segmenter
    print("Testing Semantic Segmenter...")
    
    try:
        segmenter = SemanticSegmenter()
        
        # Create a test image
        test_img = Image.new('RGB', (512, 512), color='blue')
        
        # Test segmentation
        mask, detected_objects = segmenter.segment_image(test_img)
        print(f"✅ Segmentation complete! Mask shape: {mask.shape}")
        print(f"✅ Detected objects: {detected_objects}")
        
        # Test color application
        colored = segmenter.apply_object_colors(test_img, mask, detected_objects)
        print("✅ Object color application complete!")
        
        # Test overlay
        overlay = segmenter.create_segmentation_overlay(test_img, mask)
        print("✅ Segmentation overlay complete!")
        
        print("✅ Semantic segmenter tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")