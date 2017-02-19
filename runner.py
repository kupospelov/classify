#!/usr/bin/env python3

import argparse
import logging

from classify.indexer import Indexer
from classify.loader import Loader
from classify.model import Model


def evaluation_result(prediction):
    return '{:.1f}% negative, {:.1f}% positive.'.format(
            *(round(p * 100) for p in prediction))


logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')
parser = argparse.ArgumentParser('runner')

parser.add_argument('-t', '--train',
                    dest='train',
                    metavar='TRAINING_SET',
                    action='store',
                    help='retrain the model using the train set')

parser.add_argument('-e', '--embeddings',
                    dest='embeddings',
                    action='store',
                    help='file with word embeddings',
                    required=True)

parser.add_argument('-s', '--sentences',
                    dest='sentences',
                    action='store',
                    help='process the sentences in the file')

parser.add_argument('-i', '--interactive',
                    dest='interactive',
                    action='store_true',
                    help='interactive mode')

parser.add_argument('-m', '--model',
                    dest='model',
                    action='store',
                    help='directory to store the model',
                    default='/tmp/classify.ckpt')

parser.add_argument('-l', '--length',
                    dest='length',
                    action='store',
                    help='maximum sentence length',
                    type=int,
                    default=75)

parser.add_argument('-b', '--batch-size',
                    dest='batch_size',
                    action='store',
                    help='batch size',
                    type=int,
                    default=400)

parser.add_argument('-c', '--count',
                    dest='epochs',
                    action='store',
                    help='count of epochs',
                    type=int,
                    default=50)

args = parser.parse_args()

if not args.train and not args.sentences and not args.interactive:
    print('Nothing to do.')
    exit()

indexer = Indexer(args.embeddings)
indexer.restore()
print('Embeddings loaded...')

loader = Loader(indexer)
with Model(indexer, num_hidden=75, epoch=args.epochs, max_length=args.length,
           batch_size=args.batch_size, save_path=args.model) as model:
    if args.train:
        total_input, total_output = loader.load_file(args.train)
        print('Training set loaded...')
        model.train(total_input, total_output)
    elif args.sentences or args.interactive:
        model.restore()

    if args.sentences:
        with open(args.sentences, 'r') as f:
            sentences = [line for line in f]
            tests = loader.load_sentences(sentences)
            prediction = model.predict(tests)
            for i in range(len(sentences)):
                print('{:d}) [{:s}] {:s} '.format(
                    i + 1,
                    evaluation_result(prediction[i]),
                    sentences[i].strip()))

    if args.interactive:
        print('Interactive console (empty line to stop)')
        while True:
            sentence = input('Sentence: ')
            if not sentence:
                break

            tests = loader.load_sentences([sentence])
            prediction = model.predict(tests)
            print('>>', evaluation_result(prediction[0]))
