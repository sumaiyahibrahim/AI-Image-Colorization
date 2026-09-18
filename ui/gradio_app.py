"""
Unified Gradio UI for AI Image Colorization Studio
Brings together all 6 internship tasks into one cohesive application.
"""

import gradio as gr
import numpy as np
from PIL import Image
import os
from typing import Optional, Tuple, Dict
import tempfile

# Import all our pipelines
from pipelines.base_colorizer import BaseColorizer
from pipelines.video_colorizer import VideoColorizer
from pipelines.conditional_colorizer import ConditionalColorizer
from pipelines.context_colorizer import ContextColorizer
from pipelines.historical_colorizer import HistoricalColorizer
from pipelines.cross_domain_colorizer import CrossDomainColorizer
from pipelines.historical_refinement_colorizer import HistoricalRefinementColorizer

from processing.quality_metrics import QualityMetrics
from processing.image_processor import ImageProcessor


class AIImageColorizationStudio:
    """
    Unified AI Image Colorization Studio with all 6 internship tasks.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the unified colorization studio.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"Initializing AI Image Colorization Studio")
        
        self.device = device
        self.initialized = False
        self.colorizers = {}
        self.quality_metrics = QualityMetrics()
        self.image_processor = ImageProcessor()
        
    def initialize_system(self, device_choice: str) -> str:
        """
        Initialize all colorization systems.
        
        Args:
            device_choice: Device selection from UI
        
        Returns:
            Status message
        """
        try:
            device_map = {
                "Auto (Recommended)": "auto",
                "GPU (CUDA)": "cuda",
                "CPU (Slower)": "cpu"
            }
            device = device_map.get(device_choice, "auto")
            
            print(f"Initializing all colorization systems on {device}")
            
            # Initialize all colorizers
            self.colorizers['base'] = BaseColorizer(device=device)
            self.colorizers['video'] = VideoColorizer(device=device)
            self.colorizers['conditional'] = ConditionalColorizer(device=device)
            self.colorizers['context'] = ContextColorizer(device=device)
            self.colorizers['historical'] = HistoricalColorizer(device=device)
            self.colorizers['cross_domain'] = CrossDomainColorizer(device=device)
            self.colorizers['historical_refinement'] = HistoricalRefinementColorizer(device=device)
            
            self.initialized = True
            
            status = f"""
**AI Image Colorization Studio Initialized Successfully!**

**System Status:**
• Device: {device}
• Base Colorizer: Ready
• Video Colorizer: Ready
• Conditional Colorizer: Ready
• Context-Aware Colorizer: Ready
• Historical Colorizer: Ready
• Cross-Domain Colorizer: Ready
• Historical Refinement: Ready

**Quality Metrics System:** Ready
**Image Processing System:** Ready

**All 6 internship tasks are now available!**
"""
            return status
            
        except Exception as e:
            error_msg = f"❌ **Initialization Failed:** {str(e)}"
            print(error_msg)
            return error_msg
    
    # TASK 1: Video Colorization
    def process_video_task1(
        self,
        video_input,
        segment_objects: bool,
        colorize_objects: bool,
        fps: int
    ) -> Tuple[Optional[str], str]:
        """Task 1: Process video with multi-object colorization."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if video_input is None:
            return None, "Please upload a video file!"
        
        try:
            # Create temporary output path
            output_path = os.path.join(tempfile.gettempdir(), "colorized_output.mp4")
            
            # Process video
            result = self.colorizers['video'].process_video_file(
                video_input,
                output_path,
                fps=fps,
                segment_objects=segment_objects,
                colorize_objects=colorize_objects
            )
            
            status = f"""
**Task 1: Video Processing Complete!**

**Video Information:**
• Input: {result['input_video']}
• Output: {result['output_video']}
• Original FPS: {result['original_fps']}
• Output FPS: {result['output_fps']}
• Resolution: {result['frame_width']}x{result['frame_height']}
• Frames Processed: {result['processed_frames']}/{result['total_frames']}

**Processing Options:**
• Semantic Segmentation: {result['segmentation_enabled']}
• Object Colorization: {result['colorization_enabled']}

**Segmentation Results:**
• Objects Detected: {len(result.get('segmentation_results', []))}
"""
            return output_path, status
            
        except Exception as e:
            return None, f"**Video Processing Failed:** {str(e)}"
    
    # TASK 2: Conditional Colorization
    def process_conditional_task2(
        self,
        image_input,
        conditions_text: str,
        use_segmentation: bool
    ) -> Tuple[Optional[Image.Image], str]:
        """Task 2: Conditional colorization with user-specified colors."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if image_input is None:
            return None, "Please upload an image!"
        
        try:
            # Parse color conditions
            conditions = self.colorizers['conditional'].parse_color_conditions(conditions_text)
            
            # Apply conditional colorization
            colorized, metadata = self.colorizers['conditional'].apply_color_conditions(
                image_input,
                conditions,
                use_segmentation=use_segmentation
            )
            
            # Generate quality report
            quality_report = self.quality_metrics.generate_quality_report(
                image_input, colorized, "Conditional Colorization"
            )
            quality_text = self.quality_metrics.format_quality_report(quality_report)
            
            status = f"""
**Task 2: Conditional Colorization Complete!**

**Color Conditions Applied:**
• User Input: "{conditions_text}"
• Parsed Conditions: {len(conditions)} conditions
• Conditions: {list(conditions.keys()) if conditions else 'None detected'}

**Processing Method:**
• Semantic Segmentation: {use_segmentation}
• Method: {metadata.get('method', 'unknown')}

{quality_text}
"""
            return colorized, status
            
        except Exception as e:
            return None, f"**Conditional Colorization Failed:** {str(e)}"
    
    # TASK 3: Context-Aware Colorization
    def process_context_task3(
        self,
        image_input,
        strength: float
    ) -> Tuple[Optional[Image.Image], str]:
        """Task 3: Context-aware colorization of complex scenes."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if image_input is None:
            return None, "Please upload an image!"
        
        try:
            # Apply context-aware colorization
            colorized, metadata = self.colorizers['context'].apply_context_aware_colorization(
                image_input,
                strength=strength
            )
            
            # Generate quality report
            quality_report = self.quality_metrics.generate_quality_report(
                image_input, colorized, "Context-Aware Colorization"
            )
            quality_text = self.quality_metrics.format_quality_report(quality_report)
            
            context_info = metadata.get('scene_context', {})
            
            status = f"""
**Task 3: Context-Aware Colorization Complete!**

**Scene Analysis:**
• Scene Type: {context_info.get('scene_type', 'unknown')}
• Dominant Objects: {context_info.get('dominant_objects', [])}
• Lighting: {context_info.get('lighting_estimate', 'unknown')}

**Object Relationships:**
• Relationships Analyzed: {len(context_info.get('object_relationships', []))}

**Colorization Parameters:**
• Strength: {strength}
• Method: {metadata.get('method', 'unknown')}

{quality_text}
"""
            return colorized, status
            
        except Exception as e:
            return None, f"**Context-Aware Colorization Failed:** {str(e)}"
    
    # TASK 4: Historical Colorization
    def process_historical_task4(
        self,
        image_input,
        era: str,
        manual_override: bool,
        treatment_intensity: float
    ) -> Tuple[Optional[Image.Image], str]:
        """Task 4: Time-based historical image colorization."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if image_input is None:
            return None, "Please upload an image!"
        
        try:
            # Apply historical colorization
            colorized, metadata = self.colorizers['historical'].colorize_historical_image(
                image_input,
                era=era if era else None,
                manual_override=manual_override,
                treatment_intensity=treatment_intensity
            )
            
            # Generate quality report
            quality_report = self.quality_metrics.generate_quality_report(
                image_input, colorized, "Historical Colorization"
            )
            quality_text = self.quality_metrics.format_quality_report(quality_report)
            
            status = f"""
**Task 4: Historical Colorization Complete!**

**Era Information:**
• Selected Era: {metadata.get('selected_era', 'unknown')}
• Selection Method: {metadata.get('era_selection', 'unknown')}
• Manual Override: {manual_override}

**Treatment Parameters:**
• Treatment Intensity: {treatment_intensity}
• Era Palette: {metadata.get('treatment_metadata', {}).get('palette_used', 'unknown')}

{quality_text}
"""
            return colorized, status
            
        except Exception as e:
            return None, f"**Historical Colorization Failed:** {str(e)}"
    
    # TASK 5: Cross-Domain Colorization
    def process_cross_domain_task5(
        self,
        image_input,
        domain: str,
        auto_detect: bool
    ) -> Tuple[Optional[Image.Image], str]:
        """Task 5: Cross-domain image colorization."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if image_input is None:
            return None, "Please upload an image!"
        
        try:
            # Apply cross-domain colorization
            colorized, metadata = self.colorizers['cross_domain'].colorize_cross_domain(
                image_input,
                domain=domain if domain else None,
                auto_detect=auto_detect
            )
            
            # Generate quality report
            quality_report = self.quality_metrics.generate_quality_report(
                image_input, colorized, "Cross-Domain Colorization"
            )
            quality_text = self.quality_metrics.format_quality_report(quality_report)
            
            domain_info = metadata.get('domain_info', {})
            
            status = f"""
**Task 5: Cross-Domain Colorization Complete!**

**Domain Information:**
• Selected Domain: {metadata.get('selected_domain', 'unknown')}
• Domain Name: {domain_info.get('name', 'unknown')}
• Selection Method: {metadata.get('domain_selection', 'unknown')}

**Processing:**
• Preprocessing: {metadata.get('preprocessing', {}).get('preprocessing_method', 'unknown')}
• Colorization Approach: {domain_info.get('colorization_approach', 'unknown')}

{quality_text}
"""
            return colorized, status
            
        except Exception as e:
            return None, f"**Cross-Domain Colorization Failed:** {str(e)}"
    
    # TASK 6: Historical Refinement
    def process_historical_refinement_task6(
        self,
        image_input,
        period: str,
        auto_detect_period: bool,
        refinement_intensity: float
    ) -> Tuple[Optional[Image.Image], str]:
        """Task 6: Historical photograph refinement."""
        if not self.initialized:
            return None, "Please initialize the system first!"
        
        if image_input is None:
            return None, "Please upload an image!"
        
        try:
            # Apply historical refinement colorization
            colorized, metadata = self.colorizers['historical_refinement'].colorize_historical_refined(
                image_input,
                period=period if period else None,
                auto_detect_period=auto_detect_period,
                refinement_intensity=refinement_intensity
            )
            
            # Generate quality report
            quality_report = self.quality_metrics.generate_quality_report(
                image_input, colorized, "Historical Refinement"
            )
            quality_text = self.quality_metrics.format_quality_report(quality_report)
            
            status = f"""
**Task 6: Historical Refinement Complete!**

**Refinement Pipeline:**
• Selected Period: {metadata.get('selected_period', 'unknown')}
• Selection Method: {metadata.get('period_selection', 'unknown')}
• Refinement Intensity: {refinement_intensity}

**Pipeline Stages Applied:**
• Base Colorization: Complete
• Era Treatment: Complete (Task 4 infrastructure)
• Period Refinement: Complete (Task 6 specific)
• Final Enhancement: Complete

**Facial Preservation:**
• Preservation Applied: {metadata.get('preserve_facial_features', False)}

{quality_text}
"""
            return colorized, status
            
        except Exception as e:
            return None, f"**Historical Refinement Failed:** {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the unified Gradio interface."""
        
        with gr.Blocks(
            title="AI Image Colorization Studio - ElevanceSkills Internship",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
            # AI Image Colorization Studio
            **ElevanceSkills Generative AI Internship Project**
            Complete implementation of all 6 internship tasks for image colorization
            """)
            
            # System Initialization Tab
            with gr.Tab("System Setup"):
                gr.Markdown("### Initialize the Colorization Studio")
                gr.Markdown("Initialize all AI models and processing systems before using the colorization tasks.")
                
                with gr.Row():
                    device_choice = gr.Dropdown(
                        choices=["Auto (Recommended)", "GPU (CUDA)", "CPU (Slower)"],
                        value="Auto (Recommended)",
                        label="Device Selection"
                    )
                    init_btn = gr.Button("Initialize System", variant="primary", size="lg")
                
                init_status = gr.Textbox(
                    label="System Status",
                    placeholder="Click Initialize to start the system",
                    lines=10,
                    interactive=False
                )
            
            # Task 1: Video Colorization
            with gr.Tab("Task 1: Multi-Object Video Colorization"):
                gr.Markdown("### Real-Time Multi-Object Colorization with Semantic Segmentation")
                gr.Markdown("Upload a video to colorize it with semantic segmentation and object-specific colors.")
                
                with gr.Row():
                    video_input = gr.Video(label="Upload Video")
                    video_output = gr.Video(label="Colorized Video")
                
                with gr.Row():
                    segment_objects = gr.Checkbox(True, label="Enable Semantic Segmentation")
                    colorize_objects = gr.Checkbox(True, label="Apply Object-Specific Colors")
                    fps = gr.Slider(10, 30, 15, step=1, label="Output FPS")
                
                process_video_btn = gr.Button("Process Video", variant="primary")
                video_status = gr.Textbox(label="Processing Status", lines=8, interactive=False)
            
            # Task 2: Conditional Colorization
            with gr.Tab("Task 2: Conditional Colorization"):
                gr.Markdown("### Conditional Image Colorization with User-Specified Colors")
                gr.Markdown("Specify desired colors for specific objects (e.g., 'sky:blue, grass:green, car:red')")
                
                with gr.Row():
                    cond_image_input = gr.Image(label="Upload Grayscale Image")
                    cond_image_output = gr.Image(label="Colorized Result")
                
                conditions_text = gr.Textbox(
                    label="Color Conditions",
                    placeholder="sky:blue, grass:green, car:red, building:brown",
                    lines=2
                )
                
                with gr.Row():
                    use_segmentation = gr.Checkbox(True, label="Use Semantic Segmentation")
                
                process_cond_btn = gr.Button("Apply Conditional Colors", variant="primary")
                cond_status = gr.Textbox(label="Processing Status", lines=10, interactive=False)
            
            # Task 3: Context-Aware Colorization
            with gr.Tab("Task 3: Context-Aware Colorization"):
                gr.Markdown("### Context-Aware Colorization of Complex Scenes")
                gr.Markdown("Advanced colorization that considers scene context and object relationships.")
                
                with gr.Row():
                    context_image_input = gr.Image(label="Upload Complex Scene")
                    context_image_output = gr.Image(label="Context-Aware Result")
                
                strength = gr.Slider(0.5, 1.0, 0.8, step=0.05, label="Colorization Strength")
                
                process_context_btn = gr.Button("Apply Context-Aware Colorization", variant="primary")
                context_status = gr.Textbox(label="Processing Status", lines=10, interactive=False)
            
            # Task 4: Historical Colorization
            with gr.Tab("Task 4: Historical Colorization"):
                gr.Markdown("### Time-Based Historical Image Colorization")
                gr.Markdown("Colorize historical photographs with era-specific color treatments.")
                
                with gr.Row():
                    hist_image_input = gr.Image(label="Upload Historical Photo")
                    hist_image_output = gr.Image(label="Historical Colorization Result")
                
                with gr.Row():
                    era = gr.Dropdown(
                        choices=["", "1900s", "1910s", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s"],
                        value="",
                        label="Historical Era (empty = auto-detect)"
                    )
                    manual_override = gr.Checkbox(False, label="Manual Era Override")
                
                treatment_intensity = gr.Slider(0.5, 1.0, 0.7, step=0.05, label="Era Treatment Intensity")
                
                process_hist_btn = gr.Button("Apply Historical Colorization", variant="primary")
                hist_status = gr.Textbox(label="Processing Status", lines=10, interactive=False)
            
            # Task 5: Cross-Domain Colorization
            with gr.Tab("Task 5: Cross-Domain Colorization"):
                gr.Markdown("### Cross-Domain Image Colorization")
                gr.Markdown("Colorize images from different domains: photographs, sketches, infrared, etc.")
                
                with gr.Row():
                    domain_image_input = gr.Image(label="Upload Image")
                    domain_image_output = gr.Image(label="Cross-Domain Result")
                
                with gr.Row():
                    domain = gr.Dropdown(
                        choices=["", "grayscale_photo", "sketch", "infrared", "blueprint", "newspaper", "xray"],
                        value="",
                        label="Input Domain (empty = auto-detect)"
                    )
                    auto_detect = gr.Checkbox(True, label="Auto-Detect Domain")
                
                process_domain_btn = gr.Button("Apply Cross-Domain Colorization", variant="primary")
                domain_status = gr.Textbox(label="Processing Status", lines=10, interactive=False)
            
            # Task 6: Historical Refinement
            with gr.Tab("Task 6: Historical Refinement"):
                gr.Markdown("### Historical Photograph Colorization Refinement")
                gr.Markdown("Deep refinement with period-specific treatments and facial preservation.")
                
                with gr.Row():
                    ref_image_input = gr.Image(label="Upload Historical Photo")
                    ref_image_output = gr.Image(label="Refined Colorization Result")
                
                with gr.Row():
                    period = gr.Dropdown(
                        choices=["", "1920s", "WWII", "1950s", "1960s"],
                        value="",
                        label="Historical Period (empty = auto-detect)"
                    )
                    auto_detect_period = gr.Checkbox(True, label="Auto-Detect Period")
                
                refinement_intensity = gr.Slider(0.5, 1.0, 0.8, step=0.05, label="Refinement Intensity")
                
                process_ref_btn = gr.Button("Apply Historical Refinement", variant="primary")
                ref_status = gr.Textbox(label="Processing Status", lines=10, interactive=False)
            
            # Event Handlers
            init_btn.click(
                fn=self.initialize_system,
                inputs=[device_choice],
                outputs=[init_status]
            )
            
            # Task 1 handlers
            process_video_btn.click(
                fn=self.process_video_task1,
                inputs=[video_input, segment_objects, colorize_objects, fps],
                outputs=[video_output, video_status]
            )
            
            # Task 2 handlers
            process_cond_btn.click(
                fn=self.process_conditional_task2,
                inputs=[cond_image_input, conditions_text, use_segmentation],
                outputs=[cond_image_output, cond_status]
            )
            
            # Task 3 handlers
            process_context_btn.click(
                fn=self.process_context_task3,
                inputs=[context_image_input, strength],
                outputs=[context_image_output, context_status]
            )
            
            # Task 4 handlers
            process_hist_btn.click(
                fn=self.process_historical_task4,
                inputs=[hist_image_input, era, manual_override, treatment_intensity],
                outputs=[hist_image_output, hist_status]
            )
            
            # Task 5 handlers
            process_domain_btn.click(
                fn=self.process_cross_domain_task5,
                inputs=[domain_image_input, domain, auto_detect],
                outputs=[domain_image_output, domain_status]
            )
            
            # Task 6 handlers
            process_ref_btn.click(
                fn=self.process_historical_refinement_task6,
                inputs=[ref_image_input, period, auto_detect_period, refinement_intensity],
                outputs=[ref_image_output, ref_status]
            )
        
        return interface


def main():
    """Main function to launch the application."""
    print("Starting AI Image Colorization Studio...")
    
    # Create the studio
    studio = AIImageColorizationStudio()
    
    # Create and launch the interface
    interface = studio.create_interface()
    
    print("Launching Gradio interface...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()