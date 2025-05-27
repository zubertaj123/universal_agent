#!/usr/bin/env python3
"""
Troubleshooting script for Local Call Center AI
"""
import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    print(f"Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_directory_structure():
    """Check if all required directories exist"""
    print("\n📁 Directory Structure Check")
    
    required_dirs = [
        "app",
        "app/agents",
        "app/api", 
        "app/core",
        "app/models",
        "app/services",
        "app/utils",
        "config",
        "data",
        "static",
        "templates"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ Missing: {dir_path}")
        else:
            print(f"✅ Found: {dir_path}")
    
    if missing_dirs:
        print(f"\n🔧 Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}")
    
    return len(missing_dirs) == 0

def check_init_files():
    """Check if __init__.py files exist"""
    print("\n📄 __init__.py Files Check")
    
    init_files = [
        "app/__init__.py",
        "app/agents/__init__.py",
        "app/api/__init__.py",
        "app/core/__init__.py", 
        "app/models/__init__.py",
        "app/services/__init__.py",
        "app/utils/__init__.py"
    ]
    
    missing_files = []
    for file_path in init_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n🔧 Creating missing __init__.py files...")
        for file_path in missing_files:
            Path(file_path).touch()
            print(f"   Created: {file_path}")
    
    return len(missing_files) == 0

def check_required_packages():
    """Check if required Python packages are installed"""
    print("\n📦 Package Dependencies Check")
    
    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'sqlalchemy': 'Database ORM',
        'pydantic': 'Data validation',
        'pydantic_settings': 'Settings management',
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing',
        'aiofiles': 'Async file operations',
        'python_multipart': 'File uploads',
        'jinja2': 'Template engine'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            spec = importlib.util.find_spec(package.replace('_', '-').replace('-', '_'))
            if spec is None:
                raise ImportError
            print(f"✅ {package}: {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: {description} - NOT INSTALLED")
    
    if missing_packages:
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def check_core_files():
    """Check if core application files exist"""
    print("\n🔧 Core Files Check")
    
    core_files = {
        "app/main.py": "Main application file",
        "app/core/config.py": "Configuration",
        "app/core/database.py": "Database setup",
        "app/utils/logger.py": "Logging utility",
        "requirements.txt": "Package requirements"
    }
    
    missing_files = []
    for file_path, description in core_files.items():
        if not Path(file_path).exists():
            missing_files.append((file_path, description))
            print(f"❌ Missing: {file_path} - {description}")
        else:
            print(f"✅ Found: {file_path} - {description}")
    
    return len(missing_files) == 0

def test_imports():
    """Test importing core modules"""
    print("\n🔄 Import Test")
    
    test_modules = [
        ("app.utils.logger", "Logger utility"),
        ("app.core.config", "Configuration"),
        ("app.core.database", "Database"),
    ]
    
    failed_imports = []
    for module_name, description in test_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: {description}")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            print(f"❌ {module_name}: {description} - {e}")
    
    return len(failed_imports) == 0

def create_minimal_config():
    """Create minimal configuration files"""
    print("\n⚙️ Creating minimal configuration...")
    
    # Create basic .env file
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Local Call Center AI Configuration
APP_NAME=Local Call Center AI
DEBUG=true
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./data/db/call_center.db
HOST=0.0.0.0
PORT=8000

# API Keys (replace with your actual keys)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
    
    # Create basic requirements.txt
    req_file = Path("requirements.txt")
    if not req_file.exists():
        requirements = """fastapi
uvicorn
websockets
python-dotenv
pydantic
pydantic-settings
sqlalchemy
aiosqlite
numpy
scipy
aiofiles
python-multipart
jinja2
"""
        with open(req_file, 'w') as f:
            f.write(requirements)
        print(f"✅ Created {req_file}")

def fix_metadata_issue():
    """Fix the SQLAlchemy metadata field issue"""
    print("\n🔧 Fixing SQLAlchemy metadata issue...")
    
    conversation_file = Path("app/models/conversation.py")
    if conversation_file.exists():
        try:
            content = conversation_file.read_text()
            if 'metadata = Column(' in content:
                content = content.replace('metadata = Column(', 'call_metadata = Column(')
                content = content.replace('self.metadata', 'self.call_metadata')
                content = content.replace('"metadata":', '"call_metadata":')
                
                conversation_file.write_text(content)
                print("✅ Fixed metadata field name in conversation.py")
            else:
                print("✅ metadata field already fixed or not present")
        except Exception as e:
            print(f"❌ Could not fix metadata issue: {e}")

def main():
    """Run all troubleshooting checks"""
    print("🔍 Local Call Center AI Troubleshooting")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Init Files", check_init_files),
        ("Required Packages", check_required_packages),
        ("Core Files", check_core_files),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        passed = check_func()
        if not passed:
            all_passed = False
        print()
    
    # Run fixes
    create_minimal_config()
    fix_metadata_issue()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    if all_passed and imports_ok:
        print("🎉 All checks passed! Try running the application:")
        print("   python -m app.main")
    else:
        print("⚠️  Some issues found. Please:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check the error messages above")
        print("3. Run this script again to verify fixes")
        print("4. Try the application: python -m app.main")
    
    print("\n💡 If you still have issues:")
    print("1. Make sure you're in the project root directory")
    print("2. Activate your virtual environment")
    print("3. Check Python version (3.8+ required)")
    print("4. Install packages: pip install fastapi uvicorn sqlalchemy pydantic")

if __name__ == "__main__":
    main()