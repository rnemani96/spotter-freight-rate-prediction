import logging
import sys
from pathlib import Path

# Track whether handlers have been added globally to avoid duplicates
_HANDLERS_ADDED = False

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with console + file handler.
    Console: INFO+, File: DEBUG+ written to logs/pipeline.log
    """
    global _HANDLERS_ADDED
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Let handlers decide filtering
    
    if not _HANDLERS_ADDED:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_dir / "pipeline.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Set to True so we don't add handlers to every logger multiple times
        # when we use the root logger approach or just globally.
        # Actually, adding handlers to the root logger is better to share among all,
        # but the spec asks to return a logger. Let's just make sure we don't add
        # duplicate handlers to the same logger or root.
        _HANDLERS_ADDED = True
        
        # To make it work globally, we can attach them to the root logger instead.
        logging.getLogger().addHandler(ch)
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(logging.DEBUG)
        
    return logger
