import os
import sys
from collections import deque
import logging
from utils.file_utils import get_resource_path


class SafeStreamHandler(logging.StreamHandler):
    """A safe stream handler that checks stream state before operations.
    
    This handler extends logging.StreamHandler to prevent errors when
    working with potentially closed streams.

    :ivar stream: The output stream being used
    :vartype stream: io.TextIOWrapper
    """
    def emit(self, record):
        """Emit a record if the stream is open.
        
        :param record: The log record to emit
        :type record: logging.LogRecord
        :return: None
        """
        if self.stream and not self.stream.closed:
            super().emit(record)

    def close(self):
        """Close the handler only if the stream is open.
        
        Prevents errors when closing already-closed streams.
        
        :return: None
        """
        if self.stream and not self.stream.closed:  # Only close if open
            super().close()


class BufferHandler(logging.Handler):
    """A custom logging handler that writes log messages to the buffer.
    
    This handler captures log records and writes them to an in-memory buffer.

    :cvar MAX_BUFFER_SIZE: Maximum number of lines in the buffer
    :vartype MAX_BUFFER_SIZE: int
    :cvar buffer: The deque buffer storing log messages
    :vartype buffer: collections.deque
    """
    MAX_BUFFER_SIZE = 2500  # Maximum number of lines in the buffer
    buffer = deque(maxlen=MAX_BUFFER_SIZE)

    def emit(self, record):
        """Emit a record by writing it to the buffer.

        :param record: The log record to be written
        :type record: logging.LogRecord
        :return: None
        :note: Any exceptions during emission are handled by the parent class's handleError method
        """
        try:
            msg = self.format(record)
            BufferHandler.buffer.append(msg)
        except Exception:
            self.handleError(record)

    @staticmethod
    def get_buffer_content():
        """Retrieve all content stored in the buffer.

        :return: The complete buffer contents as a single string with newline separators
        :rtype: str
        :note: The buffer maintains a fixed size (MAX_BUFFER_SIZE) and automatically
               discards oldest entries when full
        """
        return '\n'.join(BufferHandler.buffer)


class OutputHandler:
    """Handles output redirection to both logging system and original stdout/stderr streams.
    
    This class is used to intercept writes to stdout/stderr and redirect them to both
    the logging system and their original destinations. It maintains the original stream
    functionality while adding logging capabilities.
    
    :ivar level: The logging level to use for output (e.g., logging.INFO, logging.ERROR)
    :vartype level: int
    """
    
    def __init__(self, level):
        """Initialize the triple output handler.

        :param log_func: The logging function to use for output (e.g., logger.info)
        :type log_func: callable
        """
        self.level = level

    def write(self, message):
        """Process and write a message to both logging system and original stream.
        
        This method handles message formatting, filtering, and proper routing to both
        the logging system and the original stdout/stderr stream.
        
        :param message: The message to be written, which may contain multiple lines
        :type message: str
        :return: Length of the processed message
        :rtype: int
        :note: 
            - Empty messages are ignored
            - Multi-line messages are split and processed individually
            - Message length is returned to maintain stream protocol compatibility
        """
        message = message.strip()
        if not message:
            return
        for line in message.splitlines():
            logger.log(self.level, line)
        return len(message)

    def flush(self):
        """Implement stream protocol flush method.
        
        This is a no-op implementation to satisfy the stream interface requirements
        while maintaining compatibility with code that expects flush() to exist.
        
        :return: None
        :note: Actual flushing is handled by the underlying logging system and streams
        """
        pass


def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


# Define custom level
DIAGNOSE_LEVEL = 99
addLoggingLevel("DIAG", DIAGNOSE_LEVEL)

# Configure logging
if os.environ.get("FREESCRIBE_DEBUG"):
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

LOG_FORMAT = '[%(asctime)s] | %(levelname)s | %(name)s | %(threadName)s | [%(filename)s:%(lineno)d in %(funcName)s] | %(message)s'

formatter = logging.Formatter(LOG_FORMAT)

# When running a PyInstaller-built application with --windowed mode, there's no console,
# so sys.stdout and sys.stderr are set to None.
# Since Python's logging module tries to write to sys.stdout (or another stream handler),
# it fails with AttributeError: 'NoneType' object has no attribute 'write'.
if sys.stderr or sys.stdout:
    console_handler = SafeStreamHandler(sys.stderr or sys.stdout)
else:
    console_handler = logging.NullHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)

buffer_handler = BufferHandler()
buffer_handler.setLevel(LOG_LEVEL)
buffer_handler.setFormatter(formatter)

# root logger settings
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, buffer_handler],
    format=LOG_FORMAT
)
logger = logging.getLogger("freescribe")
logger.setLevel(LOG_LEVEL)

sys.stdout = OutputHandler(logging.INFO)
sys.stderr = OutputHandler(DIAGNOSE_LEVEL)
