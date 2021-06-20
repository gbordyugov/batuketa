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


def split_heads(u, num_heads):
    """Split sequence u into num_heads heads.

    Arguments:
      u: (batch_size, seq_len, d_model)-shaped input sequence.
      num_heads: number of heads. d_model should be a multiple of
      num_heads.

    Returns:
      (batch_size, num_heads, seq_len, d_model_per_head)-shaped
      tensor, where d_model_per_head = d_model // num_heads.
    """

    batch_size, seq_len, d_model = tf.shape(u)
    d_model_per_head = d_model // num_heads

    shape = tf.stack([batch_size, seq_len, num_heads, d_model_per_head])
    u = tf.reshape(u, shape)

    perm = [0, 2, 1, 3]
    u = tf.transpose(u, perm)

    return u


def merge_heads(u):
    """Merge multi-headed sequence u into a single-headed one.

    Arguments:
      u: (batch_size, num_heads, seq_len, d_model_per_head)-shaped
      input sequence.

    Returns:
      (batch_size, seq_len, d_model)-shaped tensor, where d_model =
      d_model_per_head * num_heads.
    """

    batch_size, num_heads, seq_len, d_model_per_head = tf.shape(u)
    d_model = d_model_per_head * num_heads

    perm = [0, 2, 1, 3]
    u = tf.transpose(u, perm)

    shape = tf.stack([batch_size, seq_len, d_model])
    u = tf.reshape(u, shape)

    return u


def mha(seq_len, d_model, num_heads):
    """Crete and returna multi-head attention block.

    Arguments:
      seq_len: int, the number of items in the input sequence.
      d_model: int, the dimension of the the single item representation.
      hum_heads: int, the number of heads. Note that d_model should be
        a multiple of num_heads.

    Returns:
      A Keras model with two inputs:
        input: (batch_size, seq_len, d_model)-shaped tf.int32 tensor.
        mask: (batch_size, seq_len, seq_len)-shaped tf.int32 tensor.
      and one output:
        output: (batch_size, seq_len, d_model)-shaped tf.float32 tensor.
    """

    assert (
        d_model % num_heads == 0
    ), "The dimension of the model should be a multiple of num_heads."

    input = Input(shape=(seq_len, d_model), name='input')

    Q = Dense(d_model, name='Q')
    K = Dense(d_model, name='K')
    V = Dense(d_model, name='V')

    q = split_heads(Q(input), num_heads)
    k = split_heads(K(input), num_heads)
    v = split_heads(V(input), num_heads)

    qkt = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    qkt = qkt / tf.math.sqrt(dk)

    _, _, seq_len, _ = tf.shape(qkt)

    na = tf.newaxis

    weights = Softmax(axis=-1)(qkt)

    output = tf.matmul(weights, v)

    output = merge_heads(output)

    inputs = [input]
    outputs = [q, k, v, qkt, output]

    return Model(inputs=inputs, outputs=outputs, name='mha')


def attention_model(seq_len, num_heads):
    """Create and return an attention-based summer model.

    Arguments:
      seq_len: int, the input sequence length,
      num_heads: int, the number of heads for the multi-head attention
        block.
      ffn_dim: int, the size of the hidden layer of the feed-forward
        sub-network of the multi-head attention-block.
      dropout_rate: float.

    Returns:
      A Keras model with one (batch_size, seq_len)-shaped input and
      (batch_size,)-shaped sum prediciton output.
    """

    input_ = Input(shape=(seq_len,), dtype=tf.float32, name=input_key)
    mask_ = Input(shape=(seq_len,), dtype=tf.int32, name=mask_key)

    input = input_[:,:, tf.newaxis]
    mask = mask_[:,:, tf.newaxis]

    mask = tf.cast(mask, tf.float32)

    mha_input = tf.concat([input, mask], -1)

    d_model = 2  # 1 for the numbers and 1 for the mask
    multi_head_attention = mha(seq_len, d_model, num_heads)

    _, _, _, _, mha_output = multi_head_attention(mha_input)

    mha_output = tf.reshape(mha_output, (-1, d_model*seq_len))
    sum = Dense(1, activation='relu', name='sum')(mha_output)

    sum = tf.reshape(sum, (-1,))

    inputs = [input_, mask_]
    return Model(inputs=inputs, outputs=sum)
