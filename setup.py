"""
Setup script for Surface Code Decoder project
Creates necessary directories and __init__.py files
"""

import os

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        'src',
        'src/quantum',
        'src/decoders',
        'src/models',
        'src/utils',
        'outputs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/quantum/__init__.py',
        'src/decoders/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            module_name = init_file.replace('src/', '').replace('/__init__.py', '')
            f.write(f'"""\n{module_name.capitalize()} module\n"""\n')
        print(f"✓ Created: {init_file}")

def check_dependencies():
    """Check if all required packages are installed"""
    
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'torch': 'torch',
    }
    
    missing = []
    
    print("\nChecking dependencies...")
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print("\n⚠ Missing packages. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def create_requirements_file():
    """Create requirements.txt file"""
    
    requirements = """numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
torch>=1.10.0
stim>=1.12.0
pymatching>=2.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("\n✓ Created requirements.txt")

def print_instructions():
    """Print setup instructions"""
    
    print("\n" + "="*60)
    print("  Surface Code Decoder Setup Complete!")
    print("="*60)
    print("\nProject structure created successfully.")
    print("\nTo install dependencies:")
    print("  pip install -r requirements.txt")
    print("\nTo run the simulation:")
    print("  python main.py")
    print("\nOutputs will be saved to:")
    print("  outputs/<timestamp>/")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("  Surface Code Decoder Project Setup")
    print("="*60)
    
    # Create directories
    create_directory_structure()
    
    # Create requirements file
    create_requirements_file()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Print instructions
    print_instructions()
    
    if not deps_ok:
        print("\n⚠ Please install missing dependencies before running main.py")