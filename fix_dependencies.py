#!/usr/bin/env python3
"""
Fix dependency issues for Local Call Center AI
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def fix_numpy_issue():
    """Fix NumPy compatibility issue"""
    print("🔧 Fixing NumPy compatibility issue...")
    
    # First, uninstall incompatible packages
    packages_to_reinstall = [
        "numpy",
        "scipy", 
        "torch",
        "sentence-transformers"
    ]
    
    print("1. Uninstalling problematic packages...")
    for package in packages_to_reinstall:
        run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")
    
    # Install compatible NumPy version first
    print("2. Installing NumPy 1.x...")
    if not run_command("pip install 'numpy<2.0'", "Installing NumPy 1.x"):
        return False
    
    # Install scipy with compatible version
    print("3. Installing SciPy...")
    if not run_command("pip install scipy", "Installing SciPy"):
        return False
    
    return True

def install_minimal_dependencies():
    """Install minimal set of dependencies for basic functionality"""
    print("🔧 Installing minimal dependencies...")
    
    minimal_packages = [
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "python-dotenv",
        "pydantic",
        "pydantic-settings", 
        "sqlalchemy",
        "aiosqlite",
        "aiofiles",
        "python-multipart",
        "jinja2"
    ]
    
    for package in minimal_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True

def install_optional_dependencies():
    """Install optional dependencies that might cause issues"""
    print("🔧 Installing optional dependencies...")
    
    optional_packages = [
        ("nltk", "Natural language processing"),
        ("librosa", "Audio processing"), 
        ("chromadb", "Vector database"),
        ("redis", "Caching")
    ]
    
    for package, description in optional_packages:
        print(f"Installing {package} ({description})...")
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ {package} installation failed - will be disabled in app")

def create_minimal_requirements():
    """Create a minimal requirements.txt without problematic packages"""
    minimal_requirements = """# Core FastAPI dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Database
sqlalchemy>=2.0.0
aiosqlite>=0.19.0

# File handling
aiofiles>=23.0.0
python-multipart>=0.0.6
jinja2>=3.1.0

# Basic utilities (no NumPy dependencies)
pyyaml>=6.0

# Optional - install manually if needed
# numpy<2.0  # Install this first if you need ML features
# scipy      # Install after NumPy
# nltk       # For text processing
# librosa    # For audio processing (needs NumPy)
# chromadb   # For vector database (needs NumPy)
# sentence-transformers  # For embeddings (needs PyTorch)
"""
    
    with open("requirements-minimal.txt", "w") as f:
        f.write(minimal_requirements)
    
    print("✅ Created requirements-minimal.txt")

def main():
    """Main fix function"""
    print("🔧 Fixing Local Call Center AI Dependencies")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️ Warning: Not in a virtual environment!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Please activate your virtual environment first:")
            print("  source env/bin/activate  # Linux/Mac")
            print("  env\\Scripts\\activate   # Windows")
            return
    
    print("Step 1: Creating minimal requirements file...")
    create_minimal_requirements()
    
    print("\nStep 2: Installing minimal dependencies...")
    install_minimal_dependencies()
    
    print("\nStep 3: Attempting to fix NumPy issue...")
    numpy_fixed = fix_numpy_issue()
    
    if numpy_fixed:
        print("\nStep 4: Installing optional ML dependencies...")
        install_optional_dependencies()
    else:
        print("\nStep 4: Skipping ML dependencies due to NumPy issues")
        print("You can install them manually later:")
        print("  pip install 'numpy<2.0'")
        print("  pip install scipy librosa sentence-transformers")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Test the basic app:")
    print("   python -c \"from fastapi import FastAPI; print('FastAPI works!')\"")
    print("\n2. Try running with minimal features:")
    print("   python scripts/test_minimal.py")
    print("\n3. If that works, try the full app:")
    print("   python -m app.main")
    
    if not numpy_fixed:
        print("\n⚠️ ML features (embeddings, audio processing) will be disabled")
        print("The basic call center functionality will still work!")

if __name__ == "__main__":
    main()