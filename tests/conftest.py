import pytest
import torch
from synapse.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from synapse.model.loader import load_full_model

@pytest.fixture(scope="session")
def full_model():
    torch.manual_seed(0)
    model, tokenizer = load_full_model(DEFAULT_MODEL_ID, DTYPE, DEVICE)
    model.eval()
    return model, tokenizer
