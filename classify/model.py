import logging
import tensorflow as tf
import numpy as np

from classify.lookup import Lookup


class Model:
    SCOPE_NAME = 'model'
    STATE_SIZE = 100
    DEFAULT_PATH = './model/model.ckpt'

    def __init__(self, indexer, num_hidden, epoch, max_length,
                 batch_size, save_path=DEFAULT_PATH):
        self.log = logging.getLogger('Model')
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.epoch = epoch
        self.keep_prob = tf.constant(1.0)
        self.save_path = save_path

        self.session = tf.Session()
        self.graph = self.reuse_graph()
        self.lookup = Lookup(indexer, self.max_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def train(self, inputs, outputs):
        length = len(inputs)
        self.log.debug('Training set: {:d} samples.'.format(length))

        self.session.run(tf.global_variables_initializer())
        for i in range(self.epoch):
            self.log.debug('Epoch {:3d}/{:3d} ...'.format(i + 1, self.epoch))

            errors = 0
            for blen, binp, bout in self.batch(inputs, outputs):
                vectors = self.lookup[binp]

                self.session.run(
                    self.graph['minimize'],
                    feed_dict={
                        self.graph['data']: vectors,
                        self.graph['target']: bout,
                        self.graph['lengths']: blen
                    })

                errors += self.session.run(
                    self.graph['error'],
                    feed_dict={
                        self.graph['data']: vectors,
                        self.graph['target']: bout,
                        self.graph['lengths']: blen
                    })

            self.log.debug('Errors: {:d} ({:3.1f}%)'.format(
                errors,
                100.0 * errors / length))

        saver = tf.train.Saver()
        saver.save(self.session, self.save_path)
        self.log.debug('Saved.')

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.session, self.save_path)
        self.log.debug('Restored.')

    def predict(self, tests):
        result = []

        for blen, binp, _ in self.batch(tests, []):
            vectors = self.lookup[binp]

            result.extend(self.session.run(
                self.graph['prediction'],
                feed_dict={
                    self.graph['data']: vectors,
                    self.graph['lengths']: blen
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
        lengths = tf.placeholder(tf.int32, [self.batch_size], 'lengths')

        data = tf.placeholder(
                tf.float32,
                [self.batch_size, self.max_length, Model.STATE_SIZE],
                'data')

        target = tf.placeholder(tf.float32, [self.batch_size, 2], 'target')

        cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        val, state = tf.nn.dynamic_rnn(
                cell, data, sequence_length=lengths, dtype=tf.float32)

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
