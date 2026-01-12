"""
Centralized logging utility for the project.
Handles console output with consistent formatting.
"""

import sys

class Logger:
    @staticmethod
    def header(message):
        """Print a major section header."""
        print(f"\n{'='*80}")
        print(f" {message}")
        print(f"{'='*80}")

    @staticmethod
    def subheader(message):
        """Print a clear subsection header."""
        print(f"\n{'-'*60}")
        print(f" {message}")
        print(f"{'-'*60}")

    @staticmethod
    def info(message):
        """Print a standard info message."""
        print(f"[INFO] {message}")

    @staticmethod
    def success(message):
        """Print a success message."""
        print(f"[SUCCESS] {message}")
    
    @staticmethod
    def warning(message):
        """Print a warning message."""
        print(f"[WARNING] {message}")

    @staticmethod
    def error(message):
        """Print an error message."""
        print(f"[ERROR] {message}")

    @staticmethod
    def result(key, value):
        """Print a key-value result pairing."""
        print(f"{key:<30}: {value}")
        
    @staticmethod
    def section(message):
        """Print a section divider."""
        print(f"\n[{message}]")

    @staticmethod
    def print(message):
        """Passthrough for simple prints if needed."""
        print(message)

# Global instance
log = Logger()
