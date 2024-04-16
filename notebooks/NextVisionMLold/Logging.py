import logging
from pathlib import Path 

class Logger:
    def __init__(self):
        self.log_file_path = Path("C:\out.log")        
        # Configure logging to both console and fil
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),          # Log to console
                        logging.FileHandler(self.log_file_path)  # Log to file
                    ])