#!/usr/bin/env python3

import argparse
import json
import io
import logging

from flask import Flask
from flask import request
from flask import jsonify

from classify.rest.provider import Provider
from classify.util.logger import Logger

parser = argparse.ArgumentParser('server')

parser.add_argument('-r', '--representations',
                    dest='representations',
                    action='store',
                    help='file with vector representations of words',
                    required=True)

parser.add_argument('-m', '--model-directory',
                    dest='model_directory',
                    action='store',
                    help='directory to store the models',
                    required=True)

parser.add_argument('-c', '--cache-size',
                    dest='cache_size',
                    action='store',
                    help='model cache size',
                    type=int,
                    default=3)

args = parser.parse_args()

app = Flask(__name__)
provider = Provider(
        args.representations, args.model_directory, args.cache_size)

Logger.initialize(logging.DEBUG)
log = Logger.create_with_name('Server')


@app.errorhandler(Exception)
def errorhandler(error):
    logging.exception('Error when processing query.')
    return failure(500, str(error))


def failure(code, message):
    return jsonify(message), code


def success(message='OK'):
    return jsonify(message), 200


def process_params(params):
    return json.loads(params) if params else None


@app.route('/classify/model/<modelid>', methods=['POST'])
def create_model(modelid):
    manager = provider.get_manager(modelid)

    if manager.check_path():
        return failure(403, 'Model exists')

    manager.create(process_params(request.data))
    return success()


@app.route('/classify/model/<modelid>', methods=['PUT', 'DELETE', 'GET'])
def change_model(modelid):
    manager = provider.get_manager(modelid)

    if not manager.check_path():
        return failure(404, 'Model not found')

    if request.method == 'PUT':
        manager.update(process_params(request.data))
    elif request.method == 'DELETE':
        manager.delete()
    else:
        return success(manager.read().to_dic())

    return success()


@app.route('/classify/model/<modelid>/train', methods=['POST'])
def train_model(modelid):
    if 'file' not in request.files:
        return failure(400, 'No files')

    train_file = request.files['file']
    if train_file.filename == '':
        return failure(400, 'No files selected')

    manager = provider.get_manager(modelid)
    if not manager.check_path():
        return failure(404, 'Model not found')

    data = io.StringIO(train_file.stream.read().decode('UTF8'), newline=None)
    manager.train(data)
    return success()


@app.route('/classify/model/<modelid>/run', methods=['GET'])
def run_model(modelid):
    manager = provider.get_manager(modelid)

    if not manager.check_path():
        return failure(404, 'Model not found')

    query = request.args.get('q')
    response = str(manager.predict(query))

    log.debug('Processed query "%s": %s.' % (query, response))
    return success(response)


app.run()
