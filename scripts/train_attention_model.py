from tensorflow.keras.optimizers import Adam

from batuketa.data import get_dataset
from batuketa.models import attention_model

seq_len = 100
train_n_samples = 100*1000
batch_size = 100
n_epochs = 3

train_ds = get_dataset(n_samples=train_n_samples, seq_len=seq_len)
train_ds = train_ds.batch(batch_size)

model = attention_model(seq_len=seq_len)

opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_ds, epochs=n_epochs)

eval_n_samples = 10*1000
eval_ds = get_dataset(n_samples=eval_n_samples, seq_len=seq_len)
eval_ds = eval_ds.batch(batch_size)
model.evaluate(eval_ds)
