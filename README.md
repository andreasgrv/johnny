# Johnny - DEPendency Parser

This is a work in progress.

### Status

[![Build Status](https://api.travis-ci.org/andreasgrv/johnny.svg?branch=master)](https://travis-ci.org/andreasgrv/johnny)

This is an implementation of a graph based arc factored neural dependency parser implemented using [Chainer](https://chainer.org/).
The implementation is based on the papers that can be found in the References section.

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

![Alt Text](http://johnny.overfit.xyz/parser.gif)

Python versions tested for on Debian: python2.7, python3.4, python3.5

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
