import os
import logging


# Configure logging
if os.environ.get("FREESCRIBE_DEBUG"):
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO
LOG_FILE_NAME = "../freescribe.log"
# 10 MB
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024
# Keep up to 1 backup log files
LOG_FILE_BACKUP_COUNT = 1

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE_NAME, maxBytes=LOG_FILE_MAX_SIZE, backupCount=LOG_FILE_BACKUP_COUNT)
    ]
)

logger = logging.getLogger(__name__)