import tensorflow as tf
import numpy as np

from classify.logger import Logger
from classify.lookup import Lookup
from classify.timer import Timer


class Model:
    SCOPE_NAME = 'model'
    DEFAULT_PATH = './model/model.ckpt'

    def __init__(self, indexer, params, save_path=DEFAULT_PATH):
        self.log = Logger.create(self)
        self.max_length = params.max_length
        self.batch_size = params.batch_size
        self.num_hidden = params.num_hidden
        self.keep_prob = params.keep_prob
        self.num_layers = params.num_layers
        self.epoch = params.epoch
        self.error = params.error
        self.save_path = save_path
        self.vector_dims = indexer.dimensions

        self.session = tf.Session()
        self.graph = self.reuse_graph()
        self.lookup = Lookup(indexer, self.max_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    @Timer('Training finished')
    def train(self, inputs, outputs):
        length = len(inputs)
        self.log.debug('Training set: %d samples.', length)

        self.session.run(tf.global_variables_initializer())
        for i in range(self.epoch):
            self.log.debug('Epoch %3d/%3d...', i + 1, self.epoch)

            errors = 0
            for blen, binp, bout in self.batch(inputs, outputs):
                vectors = self.lookup[binp]

                self.session.run(
                    self.graph['minimize'],
                    feed_dict={
                        self.graph['data']: vectors,
                        self.graph['target']: bout,
                        self.graph['lengths']: blen,
                        self.graph['keep_prob']: self.keep_prob
                    })

                errors += self.session.run(
                    self.graph['error'],
                    feed_dict={
                        self.graph['data']: vectors,
                        self.graph['target']: bout,
                        self.graph['lengths']: blen,
                        self.graph['keep_prob']: 1.0
                    })

            epoch_error = 100.0 * errors / length
            self.log.debug('Errors: %d (%3.1f%%)', errors, epoch_error)

            if epoch_error < self.error:
                self.log.debug('The desired accuracy achieved.')
                break

    @Timer('Saved')
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, self.save_path)

    @Timer('Restored')
    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.session, self.save_path)

    def predict(self, tests):
        result = []

        for blen, binp, _ in self.batch(tests, []):
            vectors = self.lookup[binp]

            result.extend(self.session.run(
                self.graph['prediction'],
                feed_dict={
                    self.graph['data']: vectors,
                    self.graph['lengths']: blen,
                    self.graph['keep_prob']: 1.0
                }))

        return result

    def close(self):
        self.session.close()
        self.lookup.close()
        self.log.debug('Finished.')

    def reuse_graph(self):
        with tf.variable_scope(self.SCOPE_NAME) as scope:
            return self.build_graph()
            scope.reuse_variables()

    def build_graph(self):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lengths = tf.placeholder(tf.int32, [self.batch_size], 'lengths')
        data = tf.placeholder(
                tf.float32,
                [self.batch_size, self.max_length, self.vector_dims],
                'data')
        target = tf.placeholder(tf.float32, [self.batch_size, 2], 'target')

        layers = [tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(self.num_hidden),
            output_keep_prob=keep_prob) for _ in range(self.num_layers)]
        multicell = tf.contrib.rnn.MultiRNNCell(layers)
        val, state = tf.nn.dynamic_rnn(
                multicell, data, sequence_length=lengths, dtype=tf.float32)

        # An approach to handle variable length sequences
        idx = tf.range(tf.shape(data)[0]) * self.max_length + (lengths - 1)
        last = tf.gather(tf.reshape(val, [-1, self.num_hidden]), idx)

        weight = tf.get_variable(
                'weight',
                initializer=tf.truncated_normal([self.num_hidden, 2]))

        bias = tf.get_variable('bias', initializer=tf.constant(0.1, shape=[2]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(cross_entropy)

        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        error = tf.reduce_sum(tf.cast(mistakes, tf.int32))

        return {
            'data': data,
            'target': target,
            'lengths': lengths,
            'keep_prob': keep_prob,
            'prediction': prediction,
            'minimize': minimize,
            'error': error
        }

    def batch(self, inputs, outputs):
        for i in range(0, len(inputs), self.batch_size):
            # First align the second dimension to the max sequence length
            blen = []
            binp = []
            bout = outputs[i: i + self.batch_size]

            for e in inputs[i:i + self.batch_size]:
                length = len(e)
                blen.append(length)
                binp.append(np.pad(e,
                                   ((0, self.max_length - length)),
                                   'constant').tolist())

            # Then align the first dimension to the batch size
            diff = self.batch_size - len(binp)

            if diff > 0:
                blen = np.pad(blen, ((0, diff)), 'constant').tolist()
                binp = np.pad(binp, ((0, diff), (0, 0)), 'constant').tolist()
                if outputs:
                    bout = np.pad(
                            bout, ((0, diff), (0, 0)), 'constant').tolist()

            yield blen, binp, bout
