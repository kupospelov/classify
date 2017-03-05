import logging


class Logger:
    @staticmethod
    def initialize(log_level):
        logging.basicConfig(level=log_level, format='%(name)s: %(message)s')

    @staticmethod
    def create(instance):
        return logging.getLogger(instance.__class__.__name__)
