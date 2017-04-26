# classify
An application for sentiment analysis.

## Requirements
The application runs on Linux and Windows with Python 3 and TensorFlow installed. You can use `make init` to install TensorFlow with pip.

## Usage
To run the application use `runner.py`. It requires a file with word embeddings, one word embedding per line. It can be generated with an external tool (e.g. using [fastText](https://github.com/facebookresearch/fastText)) or downloaded separately (e.g. see [GloVe](https://github.com/stanfordnlp/GloVe)).
Please run `runner.py --help` for full list of options.

### Training
Training set contains of sentences with either 1 or 0 in the end, e.g.:
```
This is an example of a positive sentence. 1
This is an example of a negative sentence. 0
```

Example:
```
$ ./runner.py -r embeddings.vec -t training_set.txt
```

### Running the model
The model can be run to evaluate sentences in a file or interactively:
```
$ ./runner.py -r embeddings.vec -s sentences.txt
$ ./runner.py -r embeddings.vec -i
```
