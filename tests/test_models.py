from math import log

import tensorflow as tf

from batuketa.models import create_history_mask
from batuketa.models import feed_forward_network
from batuketa.models import merge_heads
from batuketa.models import mha
from batuketa.models import split_heads


def test_split_heads():
    num_heads = 2
    batch_size, seq_len, d_model = 32, 20, 16
    d_model_per_head = d_model // num_heads

    input_shape = (batch_size, num_heads, seq_len, d_model_per_head)
    output_shape = (batch_size, seq_len, d_model)

    x = tf.random.uniform(shape=input_shape, dtype=tf.float32)
    y = merge_heads(x)

    assert y.shape == output_shape


def test_merge_heads():
    num_heads = 2
    batch_size, seq_len, d_model = 32, 20, 16
    d_model_per_head = d_model // num_heads

    input_shape = (batch_size, seq_len, d_model)
    output_shape = (batch_size, num_heads, seq_len, d_model_per_head)

    x = tf.random.uniform(shape=input_shape, dtype=tf.float32)
    y = split_heads(x, num_heads)

    assert y.shape == output_shape


def test_mha():
    num_heads = 2
    batch_size, seq_len, d_model = 32, 20, 16
    d_model_per_head = d_model // num_heads

    input_shape = (batch_size, seq_len, d_model)
    mask_shape = (batch_size, seq_len, seq_len)

    attention = mha(seq_len, d_model, num_heads)

    input = tf.random.uniform(shape=input_shape, dtype=tf.float32)
    mask = tf.random.uniform(  # pylint: disable=E1123
        shape=mask_shape, minval=0, maxval=2, dtype=tf.int32
    )

    q, k, v, qkt, output = attention([input, mask])

    qkv_shape = (batch_size, num_heads, seq_len, d_model_per_head)
    qkt_shape = (batch_size, num_heads, seq_len, seq_len)

    assert q.shape == qkv_shape
    assert k.shape == qkv_shape
    assert v.shape == qkv_shape
    assert qkt.shape == qkt_shape
    assert output.shape == input_shape
