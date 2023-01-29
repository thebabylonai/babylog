from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import os


def init_logger(name, dir_name="./babylog/logs/"):
    timestamp_save = datetime.now().strftime("%Y-%m-%d:%H:%M")
    daily_dir = os.path.join(dir_name, timestamp_save)

    if not (os.path.isdir(daily_dir)):
        os.makedirs(daily_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    f_handler = TimedRotatingFileHandler(
        os.path.join(daily_dir, f"{name}.log"), when="midnight", interval=1
    )
    f_handler.suffix = "%Y%m%d"
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s :%(lineno)d :  %(message)s"
    )
    f_handler.setFormatter(f_format)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s :%(lineno)d :  %(message)s"
    )
    c_handler.setFormatter(c_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


babylogger = init_logger("babylogger")
