from axyn.net.capacity import fit_layers, probe_capacity

DIMS = {"num_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
        "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064}


def test_fit_layers_bf16_more_than_fp32():
    bf16 = fit_layers(DIMS, 2, 8.0)["max_decoder_layers"]
    fp32 = fit_layers(DIMS, 4, 8.0)["max_decoder_layers"]
    assert bf16 >= 2 * fp32 - 1
    assert fp32 > 0


def test_fit_layers_caps_at_num_layers():
    r = fit_layers(DIMS, 2, 999.0)
    assert r["max_decoder_layers"] == 28
    assert r["fits_whole_model"] is True


def test_probe_capacity_shape():
    c = probe_capacity()
    assert "cpu_count" in c and c["cpu_count"] >= 1
    assert "ram_free_gb" in c and "ram_total_gb" in c
