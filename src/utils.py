from pathlib import Path
import logging
import sys

def parent_dir(d):
    """
    Return Path(../d).
    d should be a string
    """
    p_dir = Path(__file__).resolve().parents[1]
    return p_dir / d

def logging_handlers(log_filename, directory = 'logs'):
    path = parent_dir(directory)

    logfile = path / log_filename
    file_handler = logging.FileHandler(filename = logfile)
    stdout_handler = logging.StreamHandler(stream = sys.stdout)

    return [file_handler, stdout_handler]
    
