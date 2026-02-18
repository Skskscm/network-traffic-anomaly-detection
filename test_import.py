# test_import.py
print(" Testing Donut VAE imports...")

try:
    from models.donut_model import DonutVAE
    print(" DonutVAE imported successfully!")
    
    # Test creating an instance
    model = DonutVAE()
    print(" DonutVAE instance created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except ImportError as e:
    print(f" Import error: {e}")
    print("Please check models/donut_model.py file")
except Exception as e:
    print(f" Other error: {e}")