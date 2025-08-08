"""
This module provides logging utilities.
"""
import logging
import os
import time
from contextlib import contextmanager

import psutil


@contextmanager
def disable_logging():
    """
    A context manager to temporarily disable logging.
    """
    logging_disabled = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(logging_disabled)


@contextmanager
def timing(description: str):
    """
    A context manager for measuring and logging the execution duration of a code block.
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logging.info("%s took %.3f seconds.", description, duration)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    return f"RSS Memory Usage: {rss_mb:.2f} MB"