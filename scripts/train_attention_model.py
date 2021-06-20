from tensorflow.keras.optimizers import Adam

from batuketa.data import get_dataset
from batuketa.models import attention_model

seq_len = 10
n_samples = 100*1000
num_heads = 1
batch_size = 100
n_epochs = 10

ds = get_dataset(n_samples=n_samples, seq_len=seq_len)
ds = ds.batch(batch_size)

model = attention_model(seq_len=seq_len, num_heads=num_heads)

opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(ds, epochs=10)
