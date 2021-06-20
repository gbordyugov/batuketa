import tensorflow as tf
from numpy import arange
from numpy import zeros
from numpy.random import choice
from numpy.random import random
from tensorflow.data import Dataset  # pylint: disable=E0401

from batuketa.constants import input_key
from batuketa.constants import mask_key


def get_dataset(n_samples: int, seq_len: int) -> Dataset:
    """Get train/test dataset with `n_sample` samples, each having
    `seq_len` elements.

    Arguments:
      n_samples: int, number of samples.
      seq_len: int, length of sequence.

    Returns:
      Dataset yielding tuples. The first element of each tuple is a
      dictionary with keys 'input' and 'mask', and the second element
      is the sum of the masked elements of 'input'.
    """
    indices = arange(seq_len)

    def generator():
        for _ in range(n_samples):
            input = random(seq_len)
            non_zeros = choice(indices, 2, replace=False)

            sum = input[non_zeros].sum()

            mask = zeros(seq_len)
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
                input_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
                mask_key: tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
