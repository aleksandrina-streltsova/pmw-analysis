import logging
import time
from contextlib import contextmanager


@contextmanager
def disable_logging():
    logging_disabled = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(logging_disabled)


@contextmanager
def timing(description: str):
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logging.info(f"{description} took {duration:.3f} seconds.")