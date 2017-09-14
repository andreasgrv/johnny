# [![Build Status](https://api.travis-ci.org/andreasgrv/johnny.svg?branch=master)](https://travis-ci.org/andreasgrv/johnny) Johnny - DEPendency Parser

This is a work in progress.
I have uploaded the training scripts to replicate the work done in my
dissertation but I am still adding more.

### What is johnny?

This is an implementation of a graph based arc factored neural dependency parser implemented using [Chainer](https://chainer.org/). There are 3 encoders that can be used with this parser.

* Word-BILSTM, a Bidirectional LSTM encoder that encodes words.
* Char-BILSTM, a Bidirectional LSTM encoder that encodes words on the character level.
* Char-CNN, a Convolutional Neural network encoder that encodes words on the character level.

The implementation is based on the papers that can be found in the References section.

### Installation

>
	git clone https://github.com/andreasgrv/johnny
	cd johnny
	# virtualenv .env && source .env/bin/activate # optional but recommended
	pip install -r requirements.txt
	pip install .

### Training

While the basic library was tested for on Debian: python2.7, python3.4, python3.5,
the train and test utility scripts will only work on **python >= 3.3**.

Models for dependency parsing can be trained on the
[Universal Dependencies v2.0](http://universaldependencies.org/) dataset using
the **train.py** script.

Download and extract the contents to a folder of your choosing (we will refer
to this as *UD_FOLDER*, the path to the folder containing the languages).
This will probably look something like "ud-treebanks-v2.0".

To train models you can use the default blueprints I used in my dissertation. Alternatively,
if you are in for a thrill, you can override the settings to see
what happens. The blueprints can be found under the blueprints folder.

As an example, to train a parser using the Char-BILSTM encoder on the Universal Dependencies
v2.0 dataset, you can follow this snippet:

>
	mkdir models # you can use a different folder if you like
	python train.py -i UD_FOLDER -o models --verbose --name mytest \
	                --load_blueprint blueprints/dissertation/cnn-char-level.yaml
					--dataset.lang Russian # Unsurprisingly, English is the default

This will write 3 files to a directory under the models folder. The directory depends
on the name of the dataset used. The 3 files should be:

- mytest.bp (a blueprint file)   # mytest is whatever you passed to --name
- mytest.vocab (a vocabulary file)
- mytest.model (the numpy matrices of the chainer model)

You can override the defaults specified in the blueprint on the
fly from the command line using . notation. See [mlconf](https://github.com/andreasgrv/mlconf)
for details on how this works.

### Testing

In order to test a model, you need to provide the **test.py** script with the
blueprint written during training (mytest.bp if you followed the previous step).

You can test the model on the development set by providing the path to the dev
.conllu file after the --test_file option. If you want to evaluate on the
test set you need to first download it from the Universal Dependencies
[website](http://universaldependencies.org/). Make sure you provide
the test file for the right language :)

>
	python test.py --blueprint models/conll2017_v2_0/russian/mytest.bp --test_file PATH_TO_CONLLU

### Visualisation

Below is a hacky terminal visualisation of the parser predictions during training on the
[Universal Dependencies](http://universaldependencies.org/) dataset.

The white box in the Cur index row shows what word we are currently looking at in the sentence.
The number to the side is the index in the sentence - which reaches up to sentence length.
The number in |absolute value| is the distance of the real head from the current index.

For each word in the sentence the parser chooses one word to be its governor - head, to
which it draws an arc to.
This is represented by the white box in the row labelled Pred head - the height of which
roughly corresponds to the predicted probability (Can only represent few levels with a unicode box
:) ).

It then labels the predicted arc with the relationship
the two words are predicted to have as can be seen in the row labelled Pred label.

The Real head and Real label rows show which word is the correct head and label for the
training data - namely what the parser should have predicted.

![A visualisation of the parser running in the terminal](http://johnny.overfit.xyz/parser.gif)


### License

3 clause BSD, see LICENSE.txt file

### References

#### Code

[https://github.com/elikip/bist-parser](https://github.com/elikip/bist-parser)

[https://github.com/XingxingZhang/dense_parser](https://github.com/XingxingZhang/dense_parser)

#### Papers

[Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://arxiv.org/abs/1603.04351)

[Dependency Parsing as Head Selection](https://arxiv.org/abs/1606.01280)

[Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation](https://arxiv.org/abs/1508.02096)

[Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)

[From Characters to Words to in Between: Do We Capture Morphology?](https://arxiv.org/abs/1704.08352)
