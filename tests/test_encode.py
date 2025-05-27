import pytest
import torch

from detectinhos.encode import (
    decode,
    decode_landmarks,
    encode,
    encode_landmarks,
)


@pytest.fixture
def matched(encode):
    if encode == encode_landmarks:
        # Landmarks ~ [batch_size, 10]
        return torch.arange(20).reshape(2, 10).float()
    # Boxes ~ [batch_size, 4]
    return torch.tensor([[1, 1, 3, 3], [2, 2, 4, 4.0]])


@pytest.fixture
def priors(encode):
    return torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3.0]])


@pytest.fixture
def variances():
    return [0.1, 0.2]


@pytest.fixture
def encoded(encode, matched, priors, variances):
    # Call the encode function
    return encode(matched, priors, variances)


@pytest.fixture
def decoded(decode, encoded, priors, variances):
    # Call the decode function
    return decode(encoded, priors, variances)


@pytest.mark.parametrize(
    "encode, decode",
    [
        (encode, decode),
        (encode_landmarks, decode_landmarks),
    ],
)
def test_encodes_decodes(decoded, matched):
    # Assert the decoded matches the original matched data
    assert torch.allclose(decoded, matched, atol=1e-4)
