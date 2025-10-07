"""
utils/logger.py
---------------
Tiny logger factory so you can instrument modules if needed.
"""
import logging

def get_logger(name: str = "rag_project"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(ch)
    return logger