"""
AI Image Colorization Studio - Main Application Entry Point
ElevanceSkills Generative AI Internship Project

This is the main entry point for the unified application that implements
all 6 internship tasks for image colorization.

Author: ElevanceSkills Internship Candidate
Project: AI Image Colorization Studio
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ui.gradio_app import main

if __name__ == "__main__":
    print("AI Image Colorization Studio")
    print("=" * 50)
    print("ElevanceSkills Generative AI Internship Project")
    print("All 6 Tasks Implementation")
    print("=" * 50)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down AI Image Colorization Studio...")
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)