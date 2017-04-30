class Params:
    NUM_HIDDEN = 75
    NUM_LAYERS = 3
    KEEP_PROB = 0.5
    EPOCH = 50
    BATCH_SIZE = 400
    MAX_LENGTH = 100
    ERROR = 0.5

    def __init__(self):
        self.num_hidden = self.NUM_HIDDEN
        self.num_layers = self.NUM_LAYERS
        self.keep_prob = self.KEEP_PROB
        self.epoch = self.EPOCH
        self.batch_size = self.BATCH_SIZE
        self.max_length = self.MAX_LENGTH
        self.error = self.ERROR

    def fill(self, dic):
        self.num_hidden = self.get(dic, 'num_hidden', self.num_hidden)
        self.num_layers = self.get(dic, 'num_layers', self.num_layers)
        self.keep_prob = self.get(dic, 'keep_prob', self.keep_prob)
        self.epoch = self.get(dic, 'epoch', self.epoch)
        self.batch_size = self.get(dic, 'batch_size', self.batch_size)
        self.max_length = self.get(dic, 'max_length', self.max_length)
        self.error = self.get(dic, 'error', self.error)

    def to_dic(self):
        return {
                    'num_hidden' : self.num_hidden,
                    'num_layers' : self.num_layers,
                    'keep_prob' : self.keep_prob,
                    'epoch' : self.epoch,
                    'batch_size' : self.batch_size,
                    'max_length' : self.max_length,
                    'error' : self.error
                }

    @staticmethod
    def get(dic, key, default):
        # Use default value if None is explicitly stored in dic
        value = dic.get(key, default)
        return value if value is not None else default
