from pytest import approx

from batuketa.constants import input_key
from batuketa.constants import mask_key
from batuketa.data import get_dataset


def test_get_dataset():
    n_samples = 1000
    sample_length = 400

    ds = list(get_dataset(n_samples, sample_length))

    assert len(ds) == n_samples

    one_and_zero = set([0.0, 1.0])

    for inputs, target in ds:
        input = inputs[input_key]
        mask = inputs[mask_key]

        assert input.shape == (sample_length,)
        assert mask.shape == (sample_length,)
        assert target.shape == ()

        assert set(mask.numpy()) == one_and_zero

        input = input.numpy()
        mask = mask.numpy()
        target = target.numpy()

        assert mask.sum() == 2.0
        assert (input * mask).sum() == approx(target)
