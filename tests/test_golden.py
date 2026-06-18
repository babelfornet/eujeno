import pytest
import torch
from axyn.model.generate import reference_generate, pipeline_generate
from axyn.model.blocks import split_into_blocks


@pytest.mark.slow
def test_pipeline_matches_reference_generation(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids

    # 1) Riferimento: cattura PRIMA dello split (split muta i layer_idx)
    reference = reference_generate(model, ids, max_new_tokens=8)

    # 2) Pipeline distribuita in-process
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens=8)

    assert pipeline == reference, f"divergenza: {pipeline} != {reference}"
