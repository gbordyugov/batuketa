from tensorflow.keras.optimizers import Adam  # pylint: disable=E0611

from batuketa.data import get_dataset
from batuketa.models import attention_model
from batuketa.models import perfect_model

seq_len = 100
train_n_samples = 100_000
eval_n_samples = 10_000
batch_size = 100
n_epochs = 5

train_ds = get_dataset(n_samples=train_n_samples, seq_len=seq_len).cache()
train_ds = train_ds.batch(batch_size)

att_model = attention_model(seq_len=seq_len)
perf_model = perfect_model(seq_len=seq_len)

opt = Adam(learning_rate=0.01)
att_model.compile(loss='mean_squared_error', optimizer=opt)
perf_model.compile(loss='mean_squared_error', optimizer=opt)

print('Training the attention-based model:')
att_model.fit(train_ds, epochs=n_epochs)

eval_ds = get_dataset(n_samples=eval_n_samples, seq_len=seq_len).cache()
eval_ds = eval_ds.batch(batch_size)

print('Evaluating the attention-based model:')
att_model.evaluate(eval_ds)

print('Evaluating the perfect model:')
perf_model.evaluate(eval_ds)
