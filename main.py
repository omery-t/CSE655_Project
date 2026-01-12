"""
Lung Cancer Prediction - Main Entry Point

This is the main entry point for the project.
Run this file from the project root directory.

Usage:
    python main.py train              # Train and compare all models (Task 1)
    python main.py train --missing    # Train with missing values (Task 2)
    python main.py train --all        # Run both Task 1 and Task 2
    python main.py predict            # Make predictions with saved model
    python main.py predict --model svm  # Use specific model for prediction
    python main.py list               # List available saved models
    python main.py report             # View master report
    python main.py report --reset     # Reset and regenerate report
"""

import sys
import os

# Add scripts and helpers directories to path
base_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(base_dir, 'scripts')
sys.path.insert(0, scripts_dir)
sys.path.insert(0, base_dir)

from helpers.logger import log


def print_usage():
    """Print usage information."""
    log.print("""
╔══════════════════════════════════════════════════════════════════╗
║           Lung Cancer Prediction - Model Comparison              ║
╚══════════════════════════════════════════════════════════════════╝

Usage: python main.py <command> [options]

Commands:
  train              Train and compare all models (Task 1)
  train --missing    Train with synthetic missing values (Task 2)
  train --all        Run both Task 1 and Task 2
  
  predict            Make predictions using saved model
  predict --model X  Use specific model (ann, svm, xgboost, rf)
  predict --input F  Predict on custom CSV file
  
  list               List available saved models
  
  report             View master report (updated after each operation)
  report --reset     Reset report to initial state

Examples:
  python main.py train --all          # Run complete training pipeline
  python main.py predict              # Predict using best saved model
  python main.py report               # View consolidated results
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        from train import main as train_main
        
        train_args = []
        if '--missing' in sys.argv or '--missing-values' in sys.argv:
            train_args = ['--missing-values']
        elif '--all' in sys.argv:
            train_args = ['--all']
        
        sys.argv = ['train.py'] + train_args
        train_main()
        
    elif command == 'predict':
        from inference import main as inference_main
        sys.argv = ['inference.py'] + sys.argv[2:]
        inference_main()
        
    elif command == 'list':
        from inference import list_available_models
        models = list_available_models()
        
        if models:
            log.header("Available saved models")
            for m in models:
                log.print(f"  - {m}")
            log.print("-" * 30)
            log.print(f"\nTo use a model: python main.py predict --model {models[0]}")
        else:
            log.warning("No saved models found.")
            log.print("Train models first using: python main.py train")
    
    elif command == 'report':
        from helpers.report_manager import view_report, reset_report
        
        if '--reset' in sys.argv:
            log.info("Resetting master report...")
            reset_report()
            log.success("Report reset to initial state.")
        else:
            content = view_report()
            if content is None:
                log.warning("No report found. Run training first:")
                log.print("  python main.py train --all")
            
    elif command in ['-h', '--help', 'help']:
        print_usage()
        
    else:
        log.error(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
