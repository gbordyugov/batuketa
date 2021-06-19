from numpy import arange
from numpy.random import choice
from numpy.random import random
from numpy.random import zeros
from tensorflow.data import Dataset

import tensorflow as tf

from batuketa.constants import input_key
from batuketa.constants import mask_key
from batuketa.constants import target_key

def get_dataset(n_samples: int, sample_length: int) -> Dataset:
    indices = arange(sample_length)

    def generator():
        for _ in range(n_samples):
            input = random(length)
            non_zeros = choice(indices, 2, replace=False)
            sum = x[ixes].sum()
            mask = zeros(sample_length)
            mask[non_zeros] += 1.0

            yield (
                {
                    input_key: input,
                    mask_key: mask,
                },
                sum
            )

    return Dataset.from_generator(
        generator=lambda: generator,
        output_signature=(
            {
                input_key: tf.TensorSpec(shape=(sample_length,), dtype=tf.float32),
                mask_key: tf.TensorSpec(shape=(sample_length,), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )
