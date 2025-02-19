import os
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
import logging


class DualOutput:
    MAX_BUFFER_SIZE = 2500  # Maximum number of lines in the buffer
    buffer = None

    def __init__(self):
        """
        Initialize the dual output handler.

        Creates a deque buffer with a max length and stores references to original stdout/stderr streams.
        """
        DualOutput.buffer = deque(maxlen=DualOutput.MAX_BUFFER_SIZE)  # Buffer with a fixed size
        self.original_stdout = sys.stdout  # Save the original stdout
        self.original_stderr = sys.stderr  # Save the original stderr

    def write(self, message):
        """
        Write a message to the buffer, original stdout, and log file.

        :param message: The message to be written
        :type message: str
        """
        if not message.strip():
            message = '\n'
        DualOutput.buffer.append(message)

    def flush(self):
        """
        Flush the original stdout to ensure output is written immediately.
        """
        if self.original_stdout is not None:
            self.original_stdout.flush()

    @staticmethod
    def get_buffer_content():
        """
        Retrieve all content stored in the buffer.

        :return: The complete buffer contents as a single string.
        :rtype: str
        """
        return ''.join(DualOutput.buffer)


# Create a stream wrapper for stdout and stderr
class StreamToLogger:
    def __init__(self, logger, level, dual):
        self.logger = logger
        self.level = level
        self.dual = dual

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.level, message.strip())
        self.dual.write(message)

    def flush(self):
        # Required for compatibility with sys.stdout/sys.stderr
        self.dual.flush()


# Configure logging
if os.environ.get("FREESCRIBE_DEBUG"):
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO
LOG_FILE_NAME = os.path.join(os.getcwd(), "freescribe.log")
# 10 MB
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024
# Keep up to 1 backup log files
LOG_FILE_BACKUP_COUNT = 1



formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)
file_handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=LOG_FILE_MAX_SIZE, backupCount=LOG_FILE_BACKUP_COUNT)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


dual = DualOutput()
sys.stdout = StreamToLogger(logger, logging.INFO, dual)
sys.stderr = StreamToLogger(logger, logging.ERROR, dual)
