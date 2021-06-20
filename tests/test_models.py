from math import log

import tensorflow as tf

from batuketa.models import attention
from batuketa.models import create_history_mask


def test_create_history_mask():
    seq_len = 5

    seq = tf.range(seq_len)[tf.newaxis, :]

    mask = create_history_mask(seq)

    expected_mask = tf.constant(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    assert mask.shape == expected_mask.shape
    tf.debugging.assert_equal(mask, expected_mask)


def test_attention():
    batch_size, seq_len, d_model = 32, 20, 16

    input_shape = (batch_size, seq_len, d_model)
    mask_shape = (batch_size, seq_len, seq_len)

    att = attention(seq_len, d_model)

    input = tf.random.uniform(shape=input_shape, dtype=tf.float32)
    mask = tf.random.uniform(  # pylint: disable=E1123
        shape=mask_shape, minval=0, maxval=2, dtype=tf.int32
    )

    q, k, v, qkt, output = att([input, mask])

    qkv_shape = (batch_size, seq_len, d_model)
    qkt_shape = (batch_size, seq_len, seq_len)

    assert q.shape == qkv_shape
    assert k.shape == qkv_shape
    assert v.shape == qkv_shape
    assert qkt.shape == qkt_shape
    assert output.shape == input_shape
