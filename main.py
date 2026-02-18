# main.py - Unified entry point for DDoS Detection System
import argparse
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='DDoS Anomaly Detection System')
    parser.add_argument('--mode', choices=['train', 'detect', 'demo', 'results', 'analyze', 'summary'], 
                       required=True, help='Operation mode')
    parser.add_argument('--live', action='store_true', help='Enable live dashboard during training')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            print("=== Starting Training Mode ===")
            from train import train_model_with_live_dashboard
            train_model_with_live_dashboard()
        
        elif args.mode == 'detect':
            print("=== Starting Real-time Detection ===")
            from detect import real_time_detection
            real_time_detection()
        
        elif args.mode == 'demo':
            print("=== Starting Demonstration ===")
            from final_demo import live_ddos_detection_demo
            live_ddos_detection_demo()
        
        elif args.mode == 'results':
            print("=== Showing Results ===")
            from view_results import show_detection_results
            show_detection_results()
        
        elif args.mode == 'analyze':
            print("=== Analyzing Patterns ===")
            from analyze_patterns import analyze_traffic_patterns
            analyze_traffic_patterns()
        
        elif args.mode == 'summary':
            print("=== Project Summary ===")
            from project_summary import generate_summary, performance_analysis
            generate_summary()
            performance_analysis()
            
    except ImportError as e:
        print(f"Error: Could not import required module - {e}")
        print("Make sure all project files are in the same directory")
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()