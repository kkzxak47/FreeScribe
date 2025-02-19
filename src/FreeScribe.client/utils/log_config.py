import os
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
import logging


class BufferHandler(logging.Handler):
    """
    A custom logging handler that writes log messages to the TrioOutput buffer.
    """
    def emit(self, record):
        """
        Emit a record by writing it to the TrioOutput buffer.
        
        :param record: The log record to be written
        """
        try:
            msg = self.format(record)
            TrioOutput.buffer.append(msg)
        except Exception:
            self.handleError(record)


class TrioOutput:
    MAX_BUFFER_SIZE = 2500  # Maximum number of lines in the buffer
    buffer = deque(maxlen=MAX_BUFFER_SIZE)

    def __init__(self, logger, level):
        """
        Initialize the dual output handler.

        Creates a deque buffer with a max length and stores references to original stdout/stderr streams.
        """
        # DualOutput.buffer = deque(maxlen=DualOutput.MAX_BUFFER_SIZE)  # Buffer with a fixed size
        self.original_stdout = sys.__stdout__  # Save the original stdout
        self.original_stderr = sys.__stderr__  # Save the original stderr
        self.logger = logger
        self.level = level

    def write(self, message):
        """
        Write a message to the buffer, original stdout, and log file.

        :param message: The message to be written
        :type message: str
        """
        message = message.strip()
        if not message:
            return
        self.logger.log(self.level, message)
        TrioOutput.buffer.append(message)
        self.original_stdout.write(message)

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
        return '\n'.join(TrioOutput.buffer)


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

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=LOG_FILE_MAX_SIZE, backupCount=LOG_FILE_BACKUP_COUNT)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)

buffer_handler = BufferHandler()
buffer_handler.setLevel(LOG_LEVEL)
buffer_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.addHandler(buffer_handler)

trio = TrioOutput(logger, LOG_LEVEL)
sys.stdout = trio
sys.stderr = trio
