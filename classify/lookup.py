import tensorflow as tf


class Lookup:
    """Retrieves word embeddings for vector of word indices."""
    def __init__(self, indexer, max_length):
        self.max_length = max_length
        self.indexer = indexer
        self.session = tf.Session()
        self.graph = self.get_embedding_graph()

    def __getitem__(self, vectors):
        padded = self.pad(vectors)
        embeddings = self.session.run(
                self.graph['padded'],
                feed_dict={self.graph['data']: padded})

        return embeddings

    def close(self):
        self.session.close()

    def get_embedding_graph(self):
        data = tf.placeholder(tf.int32, shape=[None, None], name='data')
        embeddings = tf.constant(
                self.indexer.vectors, tf.float32, name='embeddings')

        vectors = tf.map_fn(
                lambda d: tf.nn.embedding_lookup(embeddings, d),
                data,
                tf.float32)

        padded = tf.pad(
                vectors,
                [[0, 0], [0, self.max_length - tf.shape(vectors)[1]], [0, 0]])

        return {
            'padded': padded,
            'data': data
        }

    def pad(self, vectors):
        return [v + [0] * (self.max_length - len(v)) for v in vectors]
