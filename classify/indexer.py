class Indexer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.index = 1
        self.dictionary = {}
        # vectors[0] is for unknown words
        self.vectors = [[0] * 100]

    def __len__(self):
        return self.index

    def __getitem__(self, word):
        return self.dictionary.get(word.lower(), 0)

    def restore(self):
        with open(self.file_path, 'r') as f:
            f.readline()
            for line in f:
                words = line.split()
                self.append(words[0], [float(w) for w in words[1:]])

    def append(self, word, vector):
        self.dictionary[word] = self.index
        self.vectors.append(vector)
        self.index += 1
