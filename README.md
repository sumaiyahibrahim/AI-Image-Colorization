# AI Image Colorization Studio

**ElevanceSkills Generative AI Internship Project**

A comprehensive AI-powered image colorization studio implementing all 6 internship tasks for real-time and batch image colorization using state-of-the-art generative AI techniques.

## 🎯 Project Overview

This project extends the original Stable Diffusion training project into a unified **AI Image Colorization Studio** that implements advanced image colorization techniques through multiple specialized pipelines. The application uses Stable Diffusion, semantic segmentation, and historical analysis to provide professional-grade colorization capabilities.

### ✨ Key Features

- **🎨 Unified Platform**: Single application with 6 specialized colorization modes
- **🤖 AI-Powered**: Uses Stable Diffusion and semantic segmentation models
- **📹 Video Processing**: Real-time multi-object video colorization
- **🎭 Context-Aware**: Scene understanding and object relationship analysis
- **📅 Historical Analysis**: Era detection and period-specific color treatments
- **🌐 Cross-Domain**: Support for photographs, sketches, infrared, and more
- **📊 Quality Metrics**: Comprehensive image quality evaluation system

## 🚀 Technologies Used

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **Diffusers**: HuggingFace diffusion models library
- **Transformers**: HuggingFace transformers library
- **Gradio**: Web UI framework
- **OpenCV**: Computer vision and image processing
- **PIL/Pillow**: Image manipulation
- **NumPy**: Numerical computing
- **scikit-image**: Image processing algorithms
- **torchvision**: Computer vision models (semantic segmentation)

## 📋 Installation

### Prerequisites

- Python 3.8 or higher
- GPU with CUDA support (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/aslin72/stable-diffusion-project.git
cd stable-diffusion-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python app.py
```

The application will launch a Gradio interface accessible at `http://localhost:7860`

## 🎮 Usage

### Starting the Application

1. Launch the application: `python app.py`
2. Open the provided URL in your browser
3. Click "🚀 Initialize System" in the System Setup tab
4. Choose your device (Auto/GPU/CPU) and wait for initialization
5. Navigate to any task tab to begin colorization

### Task-Specific Usage

#### Task 1: Multi-Object Video Colorization
- Upload a video file
- Enable semantic segmentation and object-specific colors
- Set output FPS and process
- View colorized video with real-time object detection

#### Task 2: Conditional Colorization
- Upload a grayscale image
- Enter color conditions (e.g., "sky:blue, grass:green, car:red")
- Enable semantic segmentation for better object detection
- Apply conditional colors and view results

#### Task 3: Context-Aware Colorization
- Upload complex scene images (cityscapes, forests, etc.)
- Adjust colorization strength
- System analyzes scene context and object relationships
- View context-aware colorization results

#### Task 4: Historical Colorization
- Upload historical photographs
- Select era (1900s-1970s) or use auto-detection
- Adjust era treatment intensity
- View historically accurate colorization

#### Task 5: Cross-Domain Colorization
- Upload images from different domains
- Select domain (photograph, sketch, infrared, etc.) or auto-detect
- System applies domain-specific preprocessing
- View domain-adapted colorization results

#### Task 6: Historical Refinement
- Upload historical photographs
- Select historical period for specialized treatment
- Adjust refinement intensity
- System applies deep period-specific refinement with facial preservation

## 🏗️ System Architecture

### Project Structure

```
stable-diffusion-project/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── colab1.ipynb                    # Original training notebook (preserved)
├── colab_cell_by_cell.py           # Original training script (preserved)
│
├── pipelines/                      # Colorization pipelines
│   ├── base_colorizer.py          # Base Stable Diffusion colorization
│   ├── video_colorizer.py         # Task 1: Video processing
│   ├── conditional_colorizer.py   # Task 2: Conditional colorization
│   ├── context_colorizer.py       # Task 3: Context-aware colorization
│   ├── historical_colorizer.py    # Task 4: Historical colorization
│   ├── cross_domain_colorizer.py  # Task 5: Cross-domain colorization
│   └── historical_refinement_colorizer.py  # Task 6: Historical refinement
│
├── segmentation/                   # Semantic segmentation
│   └── semantic_segmenter.py      # DeepLabV3-based object detection
│
├── historical/                     # Historical analysis
│   ├── era_detector.py            # Historical era detection
│   ├── historical_palette.py      # Era-specific color palettes
│   └── historical_refinement.py   # Period-specific refinement
│
├── processing/                     # Image processing utilities
│   ├── image_processor.py         # Shared image processing functions
│   └── quality_metrics.py         # Quality evaluation metrics
│
├── ui/                            # User interface
│   └── gradio_app.py              # Unified Gradio interface
│
└── outputs/                       # Generated outputs directory
```

### Pipeline Architecture

```
Input Image/Video
    ↓
Domain Detection / Era Detection / Scene Analysis
    ↓
Preprocessing (domain-specific, era-specific)
    ↓
Semantic Segmentation (object detection)
    ↓
Base Colorization (Stable Diffusion img2img)
    ↓
Conditional/Context/Historical Processing
    ↓
Quality Metrics Evaluation
    ↓
Final Output
```

## 📊 Task-wise Implementation

### Task 1: Real-Time Multi-Object Colorization with Semantic Segmentation

**Implementation**: 
- Integrated DeepLabV3 semantic segmentation model for real-time object detection
- Supports COCO dataset classes (vehicles, trees, buildings, roads, sky, people, etc.)
- Object-specific color mapping based on detected categories
- Video processing pipeline with frame-by-frame segmentation and colorization
- Gradio UI for video upload and real-time processing

**Key Features**:
- Real-time semantic segmentation using pretrained DeepLabV3
- Object category-based color schemes
- Video processing with configurable FPS
- Segmentation overlay visualization
- Batch frame extraction and processing

### Task 2: Conditional Image Colorization

**Implementation**:
- Natural language processing for parsing user color conditions
- Integration with semantic segmentation for region identification
- Color condition parser supporting multiple formats (sky:blue, grass:green)
- Object-to-color mapping system
- Preview generation for condition visualization

**Key Features**:
- Flexible condition parsing (text input)
- Semantic segmentation integration
- Color blending between user conditions and AI colorization
- Condition preview system
- Predefined color presets for common objects

### Task 3: Context-Aware Colorization of Complex Scenes

**Implementation**:
- Scene classification system (outdoor, urban, nature, indoor)
- Object relationship analysis
- Context-aware prompt generation for Stable Diffusion
- Color harmony palettes for different scene types
- Lighting condition estimation and adaptation

**Key Features**:
- Automatic scene type detection
- Object relationship mapping
- Context-specific color harmonization
- Lighting-aware colorization
- Scene visualization with context annotations

### Task 4: Time-Based Historical Image Colorization

**Implementation**:
- Historical era detection system (1900s-1970s)
- Era-specific color palettes and treatments
- Automatic era detection based on image characteristics
- Manual era override functionality
- Era-specific colorization prompts

**Key Features**:
- Automatic era detection using image analysis
- 8 historical eras with specific characteristics
- Era-specific color palettes and tone adjustments
- Manual override capability
- Historical accuracy enhancement

### Task 5: Cross-Domain Image Colorization

**Implementation**:
- Multi-domain support (grayscale photos, sketches, infrared, blueprints, newspapers, X-rays)
- Automatic domain detection using image characteristics
- Domain-specific preprocessing pipelines
- Adaptive colorization approaches per domain
- Domain visualization and comparison

**Key Features**:
- 6 input domains with specialized processing
- Automatic domain detection
- Domain-specific preprocessing (sketch enhancement, infrared normalization, etc.)
- Adaptive colorization strategies
- Cross-domain comparison capabilities

### Task 6: Historical Photograph Colorization Refinement

**Implementation**:
- Deep period-specific refinement system (1920s, WWII, 1950s, 1960s)
- Integration with Task 4's historical infrastructure
- Advanced color adjustments and texture enhancement
- Facial feature preservation system
- Period-specific artistic treatments

**Key Features**:
- 4 historical periods with specialized treatments
- Multi-stage refinement pipeline
- Facial feature preservation
- Period-specific artistic effects (vignette, soft focus, etc.)
- Integration with Task 4 era detection and palettes

## 📈 Quality Metrics

The project implements comprehensive quality evaluation metrics:

- **Saturation**: Average color saturation analysis
- **Colorfulness**: Hasler & Süsstrunk colorfulness metric
- **SSIM**: Structural Similarity Index for structural preservation
- **Edge Preservation**: Gradient correlation for edge quality
- **Skin Naturalness**: Skin region detection and naturalness evaluation
- **Overall Quality Score**: Combined metric for overall assessment

## 🔧 Configuration

### Device Selection

- **Auto**: Automatically detects and uses GPU if available
- **GPU (CUDA)**: Forces GPU usage (requires CUDA-compatible GPU)
- **CPU (Slower)**: Forces CPU usage (for systems without GPU)

### Model Settings

- **Base Model**: Stable Diffusion v1.5 (default)
- **Segmentation Model**: DeepLabV3 with ResNet50 backbone
- **Scheduler**: Euler Ancestral (default), with alternatives available

## 🚧 Limitations

- **GPU Memory**: Large models require significant GPU memory (8GB+ recommended)
- **Processing Speed**: CPU mode is significantly slower than GPU
- **Model Size**: Stable Diffusion models are large (~4GB download)
- **Video Length**: Long videos may take considerable processing time
- **Accuracy**: Automatic era/domain detection is heuristic-based and may not be 100% accurate

## 🔮 Future Improvements

- **Model Optimization**: Quantization and pruning for faster inference
- **Additional Eras**: Expand historical period coverage
- **More Domains**: Add support for additional input domains
- **Batch Processing**: Enhanced batch processing capabilities
- **API Integration**: REST API for programmatic access
- **Mobile Support**: Mobile-friendly interface and optimization
- **Cloud Deployment**: Docker containers for easy deployment

## 📝 Internship Submission

This project successfully implements all 6 ElevanceSkills internship tasks:

1. ✅ **Task 1**: Real-Time Multi-Object Colorization with Semantic Segmentation
2. ✅ **Task 2**: Conditional Image Colorization
3. ✅ **Task 3**: Context-Aware Colorization of Complex Scenes
4. ✅ **Task 4**: Time-Based Historical Image Colorization
5. ✅ **Task 5**: Cross-Domain Image Colorization
6. ✅ **Task 6**: Historical Photograph Colorization Refinement

**Submission Requirements Met**:
- ✅ All 6 tasks implemented and functional
- ✅ Built on existing training project (Stable Diffusion)
- ✅ Unified application architecture
- ✅ Comprehensive documentation
- ✅ Quality metrics system
- ✅ Professional code structure
- ✅ No unrelated datasets or projects

## 🙏 Acknowledgments

- **ElevanceSkills**: For the internship opportunity and training program
- **HuggingFace**: For Diffusers, Transformers, and model repositories
- **Stability AI**: For Stable Diffusion models
- **PyTorch Team**: For the deep learning framework
- **Gradio Team**: For the excellent UI framework

## 📄 License

This project is developed as part of the ElevanceSkills Generative AI internship program. The original Stable Diffusion models are subject to their respective licenses.

## 👤 Author

**ElevanceSkills Internship Candidate**  
B.Tech Information Technology  
Generative AI Internship Program

---

**Project Status**: ✅ Complete - All 6 tasks implemented and tested  
**Last Updated**: 2025-09-18  
**Version**: 1.0.0