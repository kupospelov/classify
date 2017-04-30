from classify.indexer import Indexer
from classify.loader import Loader
from classify.model import Model

from classify.util.cache import Cache
from classify.rest.manager import Manager


class Provider:
    def __init__(self, embeddings, model_path, cache_size):
        self.indexer = Indexer()
        self.indexer.restore(embeddings)
        self.loader = Loader(self.indexer)
        self.cache = Cache(cache_size, lambda m: m.close())
        self.path = model_path

    def get_manager(self, modelid):
        return Manager(self, modelid)

    def get_model(self, manager, restore):
        model = self.cache.get(manager.modelid)

        if not model:
            params = manager.read()
            model = Model(self.indexer, params, manager.get_model_path())

            if restore:
                model.restore()

            self.cache.set(manager.modelid, model)

        return model
