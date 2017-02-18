import re
import logging


class Loader:
    def __init__(self, indexer):
        self.indexer = indexer
        self.log = logging.getLogger('Loader')

    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            r = self.load(f)

            inputs = []
            outputs = []
            max_length = 0

            for i in r:
                max_length = max(max_length, len(i[0]))
                inputs.append(self.to_indices(i[0]))
                outputs.append(self.to_one_hot(i[1]))

            self.log.debug('Max sequence length is {:d}'.format(max_length))
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
