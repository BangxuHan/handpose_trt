import logging


def get():
    return logging.getLogger('KLAIFace_trt')


def set_level(level):
    get().setLevel(level)
