import pytest
from axyn.model.loader import model_config_dims


@pytest.mark.slow
def test_config_dims_without_loading_weights():
    dims = model_config_dims("Qwen/Qwen2.5-0.5B-Instruct")
    assert dims["num_layers"] == 24
    assert dims["hidden_size"] == 896
    assert "num_attention_heads" in dims
    assert "num_key_value_heads" in dims
