import json
import pandas as pd
import os
import numpy as np
from datetime import datetime

class ResultsManager:
    """
    Manages saving and organizing results from DDoS detection system
    """
    
    def __init__(self, base_path="results"):
        self.base_path = base_path
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for results"""
        directories = ['plots', 'metrics', 'logs']
        for directory in directories:
            path = os.path.join(self.base_path, directory)
            os.makedirs(path, exist_ok=True)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def save_training_metrics(self, metrics_dict, filename="training_metrics.json"):
        """Save training metrics to JSON file"""
        filepath = os.path.join(self.base_path, 'metrics', filename)
        
        # Add timestamp and convert to serializable types
        metrics_dict['timestamp'] = datetime.now().isoformat()
        serializable_metrics = self._convert_to_serializable(metrics_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        print(f"Training metrics saved to: {filepath}")
    
    def save_detection_results(self, results_dict, filename="detection_results.json"):
        """Save detection results to JSON file"""
        filepath = os.path.join(self.base_path, 'metrics', filename)
        
        # Add timestamp and convert to serializable types
        results_dict['timestamp'] = datetime.now().isoformat()
        serializable_results = self._convert_to_serializable(results_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Detection results saved to: {filepath}")
    
    def log_attack(self, attack_data):
        """Log detected attacks to a CSV file"""
        log_file = os.path.join(self.base_path, 'logs', 'attack_logs.csv')
        
        # Add timestamp if not present and convert to serializable
        if 'timestamp' not in attack_data:
            attack_data['timestamp'] = datetime.now().isoformat()
        
        serializable_attack = self._convert_to_serializable(attack_data)
        
        # Convert to DataFrame and save
        df = pd.DataFrame([serializable_attack])
        
        if os.path.exists(log_file):
            # Append to existing file
            existing_df = pd.read_csv(log_file)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(log_file, index=False)
        else:
            # Create new file
            df.to_csv(log_file, index=False)
        
        print(f"Attack logged")
    
    def get_recent_results(self, limit=5):
        """Get recent detection results"""
        log_file = os.path.join(self.base_path, 'logs', 'attack_logs.csv')
        
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            return df.tail(limit).to_dict('records')
        else:
            return []
    
    def generate_report(self):
        """Generate a summary report of system performance"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'directories_created': self.get_directory_structure(),
            'recent_attacks': len(self.get_recent_results()),
            'metrics_files': self.get_metrics_files()
        }
        
        report_file = os.path.join(self.base_path, 'system_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"System report generated: {report_file}")
        return report
    
    def get_directory_structure(self):
        """Get the current directory structure"""
        structure = {}
        for root, dirs, files in os.walk(self.base_path):
            current_dir = root.replace(self.base_path, '').lstrip('/')
            if current_dir == '':
                current_dir = 'root'
            structure[current_dir] = files
        
        return structure
    
    def get_metrics_files(self):
        """List all metrics files"""
        metrics_path = os.path.join(self.base_path, 'metrics')
        if os.path.exists(metrics_path):
            return os.listdir(metrics_path)
        return []