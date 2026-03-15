"""
PaperMind - Streamlit Cloud Entry Point
"""
import sys
from pathlib import Path
import os

# Add src directory to Python path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root so relative paths work
os.chdir(str(project_root))

# Execute web_ui.py with UTF-8 encoding
with open(src_path / "web_ui.py", "r", encoding="utf-8") as f:
    exec(f.read())