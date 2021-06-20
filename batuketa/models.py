import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense  # pylint: disable=E0611
from tensorflow.keras.layers import Dropout  # pylint: disable=E0611
from tensorflow.keras.layers import Input  # pylint: disable=E0611
from tensorflow.keras.layers import Softmax  # pylint: disable=E0611

from batuketa.constants import input_key
from batuketa.constants import mask_key
from batuketa.constants import output_key


def feed_forward_network(seq_len, d_model, dff):
    """Create a point-wise feed-forward network consisting of two layers.

    Args:
      d_model: The dimension of the vocabulary embedding.
      dff: The dimension of the hidden layer of the network.

    Returns:
      A keras model.
    """

    input = Input(shape=(seq_len, d_model))
    x = Dense(dff, activation='relu')(input)
    output = Dense(d_model)(x)

    return Model(inputs=input, outputs=output)


def create_history_mask(seq):
    """Return history mask.

    Args:
      seq: (batch_size, seq_len)-shaped tf.int32 tensor

    Returns:
      A (batch_size, seq_len, seq_len)-shaped tensor with elements
        a_{kij} = 1 for j > i, and a_{kij} = 0 otherwise.
    """

    seq_len = tf.shape(seq)[1]
    ones = tf.ones((seq_len, seq_len))
    mask = 1.0 - tf.linalg.band_part(ones, -1, 0)

    return mask[tf.newaxis, :, :]


def attention(seq_len, d_model):
    """Create and return a single-head attention block.

    Arguments:
      seq_len: int, the number of items in the input sequence.
      d_model: int, the dimension of the the single item representation.

    Returns:
      A Keras model with two inputs:
        input: (batch_size, seq_len, d_model)-shaped tf.float32 tensor.
        mask: (batch_size, seq_len, seq_len)-shaped tf.int32 tensor.
      and one output:
        output: (batch_size, d_model)-shaped tf.float32 tensor.
    """

    input = Input(shape=(seq_len, d_model), name='input')
    mask = Input(shape=(seq_len, seq_len), name='mask')

    Q = Dense(d_model, name='Q')
    K = Dense(d_model, name='K')
    V = Dense(d_model, name='V')

    q = Q(input)
    k = K(input)
    v = V(input)

    qkt = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    qkt = qkt / tf.math.sqrt(dk)

    infinity = 1.0e9
    qkt = qkt - infinity * mask

    weights = Softmax(axis=-1)(qkt)
    output = tf.matmul(weights, v)

    inputs = [input, mask]
    outputs = [q, k, v, qkt, output]

    return Model(inputs=inputs, outputs=outputs, name='mha')


def attention_model(seq_len):
    """Create and return an attention-based summer model.

    Arguments:
      seq_len: int, the input sequence length,

    Returns:
      A Keras model with tww (batch_size, seq_len)-shaped inputs (one
      for the input sequence and one for the mask) and
      (batch_size,)-shaped sum prediciton output.
    """

    input_ = Input(shape=(seq_len,), dtype=tf.float32, name=input_key)
    mask_ = Input(shape=(seq_len,), dtype=tf.int32, name=mask_key)

    input = input_[:, :, tf.newaxis]
    mask = mask_[:, :, tf.newaxis]

    mask = tf.cast(mask, tf.float32)

    att_input = tf.concat([input, mask], -1)

    d_model = 2  # 1 for the numbers and 1 for the mask
    att = attention(seq_len, d_model)

    history_mask = create_history_mask(input_)

    _, _, _, _, att_output = att([att_input, history_mask])

    att_output = tf.reshape(att_output, (-1, d_model * seq_len))
    sum = Dense(1, activation=None, name='sum')(att_output)

    sum = tf.reshape(sum, (-1,))

    inputs = [input_, mask_]
    return Model(inputs=inputs, outputs=sum)
