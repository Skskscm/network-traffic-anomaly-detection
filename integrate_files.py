# integrate_files.py - Check and integrate existing files
import os
import importlib.util

def check_existing_files():
    """Check what files exist and their functionality"""
    files_to_check = [
        'main.py',
        'detect.py', 
        'view_results.py',
        'analyze_patterns.py',
        'train.py',
        'test_detection.py',
        'final_demo.py'
    ]
    
    existing_files = []
    for file in files_to_check:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
    
    return existing_files

def suggest_cleanup():
    """Suggest which files to keep/merge"""
    print("\n=== FILE INTEGRATION SUGGESTIONS ===")
    
    suggestions = {
        'main.py': 'Keep as main entry point - update to use new functions',
        'detect.py': 'Update with working detection logic from final_demo.py',
        'view_results.py': 'Update to show our actual result plots',
        'analyze_patterns.py': 'Merge with test_detection.py functionality',
        'train.py': 'Keep as-is (already updated)',
        'test_detection.py': 'Keep for validation testing',
        'final_demo.py': 'Keep for demonstrations'
    }
    
    for file, suggestion in suggestions.items():
        if os.path.exists(file):
            print(f"{file}: {suggestion}")

if __name__ == "__main__":
    print("Checking project file structure...")
    existing = check_existing_files()
    suggest_cleanup()