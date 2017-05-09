import logging


class Logger:
    """Logging helpers."""
    @staticmethod
    def initialize(log_level):
        logging.basicConfig(level=log_level, format='%(name)s: %(message)s')

    @staticmethod
    def create(instance):
        return Logger.create_with_name(instance.__class__.__name__)

    @staticmethod
    def create_with_name(name):
        return logging.getLogger(name)
