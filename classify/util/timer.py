import time

from classify.util.logger import Logger


class Timer:
    """
    The decorator to output execution time of a method to debug log.
    The decorated function must be an instance method.
    """

    def __init__(self, message):
        self.message = message

    def __call__(self, fun):
        def wrapped(instance, *v, **k):
            started = time.time()
            try:
                return fun(instance, *v, **k)
            finally:
                elapsed = time.time() - started
                Logger.create(instance).debug(
                        '%s in %.2fs.', self.message, elapsed)

        return wrapped
