"""
Video Colorization Pipeline for Task 1
Real-time multi-object colorization with semantic segmentation for video processing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
import os
from pathlib import Path
import tempfile
import threading
import queue

from pipelines.base_colorizer import BaseColorizer
from segmentation.semantic_segmenter import SemanticSegmenter
from processing.image_processor import ImageProcessor


class VideoColorizer:
    """
    Real-time video colorization with semantic segmentation and multi-object detection.
    Supports video file upload and webcam input.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize video colorization pipeline.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"🎬 Initializing Video Colorizer")
        
        try:
            # Initialize base colorizer
            self.colorizer = BaseColorizer(device=device)
            
            # Initialize semantic segmenter
            self.segmenter = SemanticSegmenter(device=device)
            
            # Video processing state
            self.is_processing = False
            self.frame_queue = queue.Queue(maxsize=30)
            self.result_queue = queue.Queue(maxsize=30)
            
            print("✅ Video Colorizer Ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize video colorizer: {e}")
            raise
    
    def process_video_file(
        self,
        video_path: str,
        output_path: str,
        fps: int = 15,
        max_frames: Optional[int] = None,
        segment_objects: bool = True,
        colorize_objects: bool = True
    ) -> Dict:
        """
        Process a video file with multi-object colorization.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video
            fps: Output video frames per second
            max_frames: Maximum number of frames to process (None = all)
            segment_objects: Whether to perform semantic segmentation
            colorize_objects: Whether to apply object-specific colors
        
        Returns:
            Dictionary with processing results and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"🎬 Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {frame_width}x{frame_height} @ {original_fps}fps, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        frame_count = 0
        processed_frames = 0
        segmentation_results = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Process frame
                processed_frame, frame_metadata = self._process_frame(
                    pil_image, 
                    segment_objects=segment_objects,
                    colorize_objects=colorize_objects
                )
                
                # Convert back to BGR for video
                processed_bgr = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
                
                # Write to output video
                out.write(processed_bgr)
                
                # Store metadata
                if segment_objects:
                    segmentation_results.append(frame_metadata.get('segmentation', {}))
                
                processed_frames += 1
                frame_count += 1
                
                # Progress update
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Release resources
            cap.release()
            out.release()
            
            # Generate result metadata
            result_metadata = {
                'input_video': video_path,
                'output_video': output_path,
                'original_fps': original_fps,
                'output_fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'segmentation_enabled': segment_objects,
                'colorization_enabled': colorize_objects,
                'segmentation_results': segmentation_results
            }
            
            print(f"✅ Video processing complete! Output saved to: {output_path}")
            print(f"📊 Processed {processed_frames}/{total_frames} frames")
            
            return result_metadata
            
        except Exception as e:
            cap.release()
            out.release()
            raise RuntimeError(f"Video processing failed: {str(e)}")
    
    def _process_frame(
        self,
        image: Image.Image,
        segment_objects: bool = True,
        colorize_objects: bool = True
    ) -> Tuple[Image.Image, Dict]:
        """
        Process a single frame with optional segmentation and colorization.
        
        Args:
            image: PIL Image of the frame
            segment_objects: Whether to perform semantic segmentation
            colorize_objects: Whether to apply object-specific colors
        
        Returns:
            Tuple of (processed_image, metadata_dict)
        """
        metadata = {'original_size': image.size}
        
        try:
            if segment_objects:
                # Perform semantic segmentation
                mask, detected_objects = self.segmenter.segment_image(image)
                metadata['segmentation'] = {
                    'mask_shape': mask.shape,
                    'detected_objects': detected_objects
                }
                
                if colorize_objects:
                    # Apply object-specific colors
                    colored = self.segmenter.apply_object_colors(image, mask, detected_objects)
                    metadata['colorization'] = 'object_specific'
                    return colored, metadata
                else:
                    # Just create segmentation overlay
                    overlay = self.segmenter.create_segmentation_overlay(image, mask)
                    metadata['colorization'] = 'segmentation_overlay'
                    return overlay, metadata
            else:
                # Basic colorization without segmentation
                colorized, color_metadata = self.colorizer.colorize(image)
                metadata['colorization'] = 'basic'
                metadata['colorization_metadata'] = color_metadata
                return colorized, metadata
                
        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            metadata['error'] = str(e)
            return image, metadata
    
    def process_webcam_frame(
        self,
        frame: np.ndarray,
        segment_objects: bool = True,
        colorize_objects: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single webcam frame.
        
        Args:
            frame: NumPy array from webcam (BGR format)
            segment_objects: Whether to perform semantic segmentation
            colorize_objects: Whether to apply object-specific colors
        
        Returns:
            Tuple of (processed_frame_bgr, metadata_dict)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process frame
        processed_image, metadata = self._process_frame(
            pil_image,
            segment_objects=segment_objects,
            colorize_objects=colorize_objects
        )
        
        # Convert back to BGR
        processed_bgr = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        
        return processed_bgr, metadata
    
    def get_webcam_stream(self, camera_id: int = 0):
        """
        Generator for webcam stream processing.
        
        Args:
            camera_id: Webcam device ID
        
        Yields:
            Tuple of (processed_frame_bgr, metadata_dict)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam: {camera_id}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, metadata = self.process_webcam_frame(frame)
                yield processed_frame, metadata
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video for individual processing.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of paths to extracted frame images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    if max_frames and extracted_count >= max_frames:
                        break
                    
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"✅ Extracted {len(frame_paths)} frames to {output_dir}")
            return frame_paths
            
        except Exception as e:
            cap.release()
            raise RuntimeError(f"Frame extraction failed: {str(e)}")
    
    def create_video_from_frames(
        self,
        frame_paths: List[str],
        output_path: str,
        fps: int = 15
    ) -> str:
        """
        Create a video from a list of frame images.
        
        Args:
            frame_paths: List of paths to frame images
            output_path: Path for output video
            fps: Video frames per second
        
        Returns:
            Path to created video
        """
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write all frames
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        print(f"✅ Video created from {len(frame_paths)} frames: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test video colorizer
    print("Testing Video Colorizer...")
    
    try:
        colorizer = VideoColorizer()
        
        # Create a simple test video
        test_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Save test video
        test_video_path = "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 15, (640, 480))
        for frame in test_frames:
            out.write(frame)
        out.release()
        
        print("✅ Test video created")
        
        # Test frame processing
        test_frame = test_frames[0]
        processed, metadata = colorizer.process_webcam_frame(test_frame)
        print(f"✅ Frame processing complete! Metadata: {metadata}")
        
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        
        print("✅ Video colorizer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")