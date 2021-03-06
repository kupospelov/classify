import re

from classify.util.logger import Logger
from classify.util.timer import Timer


class Loader:
    """Loads input and output data for use in model."""
    def __init__(self, indexer):
        self.log = Logger.create(self)
        self.indexer = indexer

    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            return self.load_data(f)

    @Timer('Training set loaded')
    def load_data(self, data):
        r = self.load(data)

        inputs = []
        outputs = []
        max_length = 0

        for i in r:
            max_length = max(max_length, len(i[0]))
            inputs.append(self.to_indices(i[0]))
            outputs.append(self.to_one_hot(i[1]))

        self.log.debug('Max sequence length is %d.', max_length)
        return inputs, outputs

    def load_sentences(self, sentences):
        return [self.to_indices(self.handle_line(s)) for s in sentences]

    def load(self, source):
        return [(words[:-1], int(words[-1]))
                for words in (self.handle_line(line) for line in source)]

    def to_indices(self, words):
        return [self.indexer[w] for w in words]

    def to_one_hot(self, value):
        result = [0, 0]
        result[value] = 1
        return result

    def handle_line(self, line):
        return re.findall(r'[\w]+', line)
