import pytest
import torch
from synapse.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from synapse.model.loader import load_full_model


@pytest.fixture(scope="session")
def _full_model_session():
    torch.manual_seed(0)
    model, tokenizer = load_full_model(DEFAULT_MODEL_ID, DTYPE, DEVICE)
    model.eval()
    return model, tokenizer


@pytest.fixture
def full_model(_full_model_session):
    """Carica il modello una sola volta per sessione, ma garantisce l'ISOLAMENTO
    fra test: split_into_blocks muta in place layer.self_attn.layer_idx, quindi
    un test che splitta corromperebbe il modello condiviso per i test successivi
    (es. reference_generate). Salviamo gli indici globali originali e li
    ripristiniamo dopo ogni test."""
    model, tokenizer = _full_model_session
    original_layer_idx = [layer.self_attn.layer_idx for layer in model.model.layers]
    yield model, tokenizer
    for layer, idx in zip(model.model.layers, original_layer_idx):
        layer.self_attn.layer_idx = idx
