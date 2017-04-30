import json
import os
import shutil

from classify.params import Params


class Manager:
    def __init__(self, provider, modelid):
        self.provider = provider
        self.modelid = modelid
        self.path = self.get_path(modelid)

    def read(self):
        config_path = self.get_config_path()
        with open(config_path, 'r') as f:
            config = f.read()
            stored = json.loads(config)
            params = Params()
            params.fill(stored)

        return params

    def create(self, dic):
        params = self.prepare_params(Params(), dic)
        self.create_directories()
        self.write(params)

    def update(self, dic):
        params = self.prepare_params(self.read(), dic)
        self.write(params)

    def delete(self):
        self.provider.cache.pop(self.modelid)
        shutil.rmtree(self.path)

    def train(self, training_set):
        totalinput, totaloutput = self.provider.loader.load_data(training_set)
        model = self.provider.get_model(self, False)
        model.train(totalinput, totaloutput)
        model.save()

    def predict(self, sentence):
        tests = self.provider.loader.load_sentences([sentence])
        model = self.provider.get_model(self, True)
        prediction = model.predict(tests)
        return prediction[0]

    def check_path(self):
        return os.path.exists(self.path)

    def get_model_path(self):
        return os.path.join(self.path, 'model.ckpt')

    def write(self, params):
        stored = json.dumps(params.to_dic())
        config_path = self.get_config_path()
        with open(config_path, 'w') as f:
            f.write(stored)

    def create_directories(self):
        os.makedirs(self.path)

    def get_path(self, modelid):
        return os.path.join(self.provider.path, modelid)

    def get_config_path(self):
        return os.path.join(self.path, 'config.json')

    def prepare_params(self, params, dic):
        if dic:
            params.fill(dic)

        return params
