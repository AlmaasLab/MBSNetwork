from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from queue import Queue


def configure_root_logger(log_file_path: Path):
    root = logging.getLogger()

    # Check if handlers already exist to avoid duplicates
    if not root.handlers:
        console_handler = logging.StreamHandler()

        try:
            file_handler = logging.FileHandler(log_file_path)
        except OSError as e:
            root.error(f"Failed to open log file {log_file_path}: {e}")
            return root

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.setLevel(logging.INFO)
    return root


def configure_worker_logger(log_queue: Queue, log_level: str):
    """Need to configure a separate logger for the worker process as the logger
    is not propagated to child processes."""
    worker_logger = logging.getLogger("worker")
    if not worker_logger.hasHandlers():
        handler = QueueHandler(log_queue)
        worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)
    return worker_logger


def configure_queue_listener(
    root_logger: logging.Logger,
) -> tuple[Queue, QueueListener]:
    """Configure a queue listener to handle logging messages from worker processes."""
    manager = Manager()
    queue = manager.Queue()
    listener = QueueListener(queue, *root_logger.handlers)
    return queue, listener
