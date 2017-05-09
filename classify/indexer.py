from classify.util.logger import Logger
from classify.util.timer import Timer


class Indexer:
    """Determines the index of a word in the word embedding matrix."""
    def __init__(self):
        self.log = Logger.create(self)
        self.dictionary = {}
        self.vectors = []
        self.dimensions = None

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, word):
        return self.dictionary.get(word.lower(), 0)

    @Timer('Word embeddings loaded')
    def restore(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                words = line.split()
                self.append(words[0], [float(w) for w in words[1:]])

    def append(self, word, vector):
        self.setdim(len(vector))
        self.dictionary[word] = len(self.vectors)
        self.vectors.append(vector)

    def setdim(self, value):
        if self.dimensions is None:
            # vectors[0] is for unknown words
            self.dimensions = value
            self.vectors.append([0] * value)
            self.log.debug('Vector size set to %d.', value)
        else:
            if self.dimensions != value:
                raise ValueError(
                        'Attempt to set a different value for '
                        'previously initialized dimension')
