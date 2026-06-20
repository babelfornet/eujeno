import pytest
from eujeno.model.loader import model_dims

@pytest.mark.slow
def test_loads_with_expected_dims(full_model):
    model, tokenizer = full_model
    dims = model_dims(model)
    assert dims["num_layers"] == 24
    assert dims["hidden_size"] == 896
    assert tokenizer is not None
