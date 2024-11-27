import os
import shutil
import logging
from .logger import get, set_level


def _init_log():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    from logging.handlers import TimedRotatingFileHandler
    logger = logging.getLogger('KLAIFace_trt')
    handler = TimedRotatingFileHandler(filename=os.path.join('./logs', 'log.log'), when='MIDNIGHT', backupCount=2)
    console_handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s][%(levelname)s] | [PID]%(process)d[TID]%(thread)d | [FILE]%(filename)s[LINE]%(lineno)d | [%(funcName)s]%(message)s")
    handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(handler)
    # logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


if __name__ == 'logger':
    _init_log()
