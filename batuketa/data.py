import tensorflow as tf
from numpy import arange
from numpy import zeros
from numpy.random import choice
from numpy.random import random
from tensorflow.data import Dataset  # pylint: disable=E0401

from batuketa.constants import input_key
from batuketa.constants import mask_key


def get_dataset(n_samples: int, sample_length: int) -> Dataset:
    indices = arange(sample_length)

    def generator():
        for _ in range(n_samples):
            input = random(sample_length)
            non_zeros = choice(indices, 2, replace=False)

            sum = input[non_zeros].sum()

            mask = zeros(sample_length)
            mask[non_zeros] += 1.0

            yield (
                {
                    input_key: input,
                    mask_key: mask,
                },
                sum,
            )

    return Dataset.from_generator(
        generator=generator,
        output_signature=(
            {
                input_key: tf.TensorSpec(
                    shape=(sample_length,), dtype=tf.float32
                ),
                mask_key: tf.TensorSpec(
                    shape=(sample_length,), dtype=tf.float32
                ),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
