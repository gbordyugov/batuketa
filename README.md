# Project Batuketa

![build status](https://github.com/gbordyugov/batuketa/actions/workflows/build-3.7.yml/badge.svg)
![build status](https://github.com/gbordyugov/batuketa/actions/workflows/build-3.8.yml/badge.svg)

## Installing and running the code

This project requires Python 3.7 or 3.8. The models are neural
networks coded in Tensorflow 2.5.0.

Batuketa uses [poetry](https://python-poetry.org/) as a build tool and
dependency manager. Poetry takes care of creating, maintaining, and
activating/deactivating a dedicated virtual environment for the
project without you having to do it. Once the virtual environment has
been created by poetry (which typicall happens on a first run of
`poetry install`), any command prefixed by "poetry run" will be
automagically executed in this virtual environment.

Those steps are needed to build and install the package, and to run a
sample training/evalution script:

1. Clone repository and `cd` into it.
1. Make sure you've got the latest version of the `poetry` tool
   available in your Python installation, otherwise install it by
   issuing `pip install --upgrade poetry`.
1. Install the project as a Python module along with its dependencies
   by issuing `poetry install` (make sure you're in the directory with
   the cloned git repository of the project).
1. Run unit tests by issuing `poetry run pytest`.
1. Run model training and evaluation by issuing `poetry run python
   scripts/train_attention_model.py`. A typical output would look like
```bash
➜  batuketa git:(master) ✗ poetry run python scripts/train_attention_model.py
Training the attention-based model:
Epoch 1/5
1000/1000 [==============================] - 48s 48ms/step - loss: 0.0304
Epoch 2/5
1000/1000 [==============================] - 17s 17ms/step - loss: 0.0010
Epoch 3/5
1000/1000 [==============================] - 17s 17ms/step - loss: 3.2781e-04
Epoch 4/5
1000/1000 [==============================] - 21s 21ms/step - loss: 1.5733e-04
Epoch 5/5
1000/1000 [==============================] - 17s 17ms/step - loss: 1.0748e-04
Evaluating the attention-based model:
100/100 [==============================] - 5s 48ms/step - loss: 4.3330e-06
Evaluating the perfect model:
100/100 [==============================] - 0s 1ms/step - loss: 1.7936e-15
```

In the above step, an attention-based model is trained for five epochs
on the training dataset, then it is evaluated against the eval
dataset. Additional, the "perfect" model is evaluated on the same eval
dataset for the sake of performance comparison.

The file
[scripts/train_attention_model.py](scripts/train_attention_model.py)
represents a short, user-friendly driver script with a few
hyperparameters that can be changed in order to evaluate the
performance of the model for different settings.


## Choice of models

### The attention-based model

As the primary working model, I implemented a single-headed
[attention](https://arxiv.org/abs/1706.03762)-based network. The main
idea behind the attention technique is to let the model learn how to
attend to different elements of the input sequence depending on their
values.

A crucial intermediate product of the attention architecture is a
square attention matrix of the size given by the sequence length. This
attention matrix encodes the relative importance of the i-th element
of the sequence to the j-th one for all i, j from zero to sequence
length.

Another important component of the attention-based model is the
so-called history mask. This attention masks makes sure that, for each
position in the input sequence, the model takes into account the
influence only of the previous elements, thus ignoring the "unknown"
future.

One performance drawback of the model is its square algorithmic
complexity in the input length.

For more details on the method, please see the [original
paper](https://arxiv.org/abs/1706.03762). I would be also glad to
discuss the details of the approach in person.


### The pefect model

I also implemented the "perfect" model, which is a non-trainable
Tensorflow computational graph that calculates the desired sum of two
elements by design. As it has zero trainable parameters, it's
perfomance with the task is perfect (up to the numerical errors). I
use it as an ideal model to benchmark the attention-based model
against.


## Training and evaluation data

The file [batuketa/data.py](batuketa/data.py) contains the code for
lazy generation of Tensorflow Datasets that are used for both training
and model evaluation.


## Code quality assurance

There are two independent Github Actions build scripts in [the
corresponding directory](.github/workflows) for Python versions 3.7
and 3.8.

To run unit tests, issue `poetry run pytest` (after you have installed
the project along with its dependencies as described above).

The scripts in the [qa](qa/) directory perform formatting checks using
the [black](https://github.com/psf/black) tool, import sorting with
the help of [isort](https://github.com/PyCQA/isort), and static code
analysis with the help of [pylint](https://www.pylint.org/).
