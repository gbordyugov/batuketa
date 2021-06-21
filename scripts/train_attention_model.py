import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard  # pylint: disable=E0611
from tensorflow.keras.optimizers import Adam  # pylint: disable=E0611

from batuketa.data import get_dataset
from batuketa.models import attention_model
from batuketa.models import perfect_model

# Length of the input sequence
seq_len = 100

# Number of samples in training dataset
train_n_samples = 100_000

# Number of samples in evaluation dataset
eval_n_samples = 10_000

# Those are self-explanatory
batch_size = 100
n_epochs = 10

# TensorBoard logs
log_dir = 'logs/'

train_ds = (
    get_dataset(n_samples=train_n_samples, seq_len=seq_len)
    .cache()
    .shuffle(10 * batch_size)
)
train_ds = train_ds.batch(batch_size)

eval_ds = get_dataset(n_samples=eval_n_samples, seq_len=seq_len).cache()
eval_ds = eval_ds.batch(batch_size)

att_model = attention_model(seq_len=seq_len)
perf_model = perfect_model(seq_len=seq_len)

opt = Adam(learning_rate=0.01)
att_model.compile(loss='mse', optimizer=opt)
perf_model.compile(loss='mse', optimizer=opt)

tensorboard_callback = TensorBoard(log_dir=log_dir)

print('Training the attention-based model:')
att_model.fit(train_ds, epochs=n_epochs, callbacks=[tensorboard_callback])


print('Evaluating the attention-based model:')
att_model.evaluate(eval_ds)

print('Evaluating the perfect model:')
perf_model.evaluate(eval_ds)
