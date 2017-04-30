import collections

from classify.logger import Logger


class Cache:
    def __init__(self, size, evict_action):
        self.log = Logger.create(self)
        self.size = size
        self.cache = collections.OrderedDict()
        self.action = evict_action

    def get(self, key, value=None):
        found = self.cache.get(key, None)

        if found:
            self.cache.move_to_end(key)
            self.log.debug('Value refreshed: %s.' % key)
        else:
            if value:
                self.set(key, value)
                return value

        return found

    def set(self, key, value):
        if not self.pop(key):
            if len(self.cache) >= self.size:
                item = self.cache.popitem(last=False)
                self.log.debug('Least used key removed: %s.' % item[0])
                self.action(item[1])

        self.cache[key] = value
        self.log.debug('Value added: %s.' % key)

    def pop(self, key):
        found = self.cache.pop(key, None)

        if found:
            self.action(found)

        return found

    def __del__(self):
        try:
            while True:
                item = self.cache.popitem()
                self.action(item[1])
        except KeyError:
            pass
