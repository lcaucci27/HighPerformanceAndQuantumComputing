"""
Setup Verification Script
Checks that all required packages are installed and working
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} - MISSING")
        print(f"  Error: {e}")
        return False

def main():
    """Run verification checks"""
    print("="*60)
    print(" Surface Code Decoder - Setup Verification")
    print("="*60)
    print()
    
    print("Checking required packages:")
    print("-"*60)
    
    packages = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("stim", "stim"),
        ("pymatching", "pymatching"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("PIL", "PIL"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    print("-"*60)
    print()
    
    if all_ok:
        print("✓ All required packages are installed!")
        print()
        
        # Check versions
        print("Package versions:")
        print("-"*60)
        import numpy
        import torch
        import stim
        import pymatching
        import matplotlib
        import scipy
        
        print(f"  numpy:      {numpy.__version__}")
        print(f"  torch:      {torch.__version__}")
        print(f"  stim:       {stim.__version__}")
        print(f"  pymatching: {pymatching.__version__}")
        print(f"  matplotlib: {matplotlib.__version__}")
        print(f"  scipy:      {scipy.__version__}")
        print("-"*60)
        print()
        
        # Check PyTorch device
        print("PyTorch configuration:")
        print("-"*60)
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"  CPU threads:    {torch.get_num_threads()}")
        print("-"*60)
        print()
        
        # Quick functionality test
        print("Running quick functionality test...")
        print("-"*60)
        
        try:
            # Test Stim
            import stim
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=3,
                distance=3,
                after_clifford_depolarization=0.01
            )
            print(f"✓ Stim: Generated d=3 circuit ({circuit.num_qubits} qubits)")
            
            # Test PyMatching
            import pymatching
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            print(f"✓ PyMatching: Created matcher ({circuit.num_detectors} detectors)")
            
            # Test torch
            import torch
            x = torch.randn(10, 20)
            y = torch.nn.Linear(20, 5)(x)
            print(f"✓ PyTorch: Neural network forward pass successful")
            
            print("-"*60)
            print()
            print("✓ All functionality tests passed!")
            print()
            print("="*60)
            print(" Setup is complete. You can now run: python main.py")
            print("="*60)
            
        except Exception as e:
            print(f"✗ Functionality test failed: {e}")
            print()
            print("Please check your installation.")
            return 1
        
        return 0
        
    else:
        print("✗ Some packages are missing!")
        print()
        print("Please install missing packages using:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install individually:")
        print("  pip install numpy torch stim pymatching matplotlib scipy Pillow")
        return 1

if __name__ == "__main__":
    sys.exit(main())