import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_full_model(model_id: str, dtype: torch.dtype, device: str):
    """Carica modello completo + tokenizer. Usato come riferimento e come
    sorgente da cui estrarre i blocchi (Task 5)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def model_dims(model) -> dict:
    cfg = model.config
    return {
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    }


def model_config_dims(model_id: str) -> dict:
    """Dimensioni del modello dalla sola AutoConfig (NIENTE pesi scaricati)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return {
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "model_type": cfg.model_type,
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
    }


def load_partial_model(model_id: str, stages, dtype, device):
    """Carica in RAM SOLO i pesi dei layer assegnati (+ embed/head se nei tuoi stage);
    il resto del modello resta su 'meta' (zero memoria). Ritorna (model, tokenizer)."""
    import glob
    import os

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    tie = bool(getattr(config, "tie_word_embeddings", False))
    prefixes = set()
    if stages.embed or (stages.head and tie):
        prefixes.add("model.embed_tokens.")
    if stages.head:
        prefixes.add("model.norm.")
        if not tie:
            prefixes.add("lm_head.")
    for (lo, hi) in stages.decoders:
        for i in range(lo, hi):
            prefixes.add(f"model.layers.{i}.")

    path = snapshot_download(model_id)
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if any(key.startswith(p) for p in prefixes):
                    set_module_tensor_to_device(model, key, device, value=sf.get_tensor(key).to(dtype))

    # la rotary embedding ha un buffer inv_freq calcolato a init: con init_empty_weights
    # finisce su meta, quindi va ri-materializzata sui nodi che servono decoder.
    if stages.decoders:
        model.model.rotary_emb = type(model.model.rotary_emb)(config=config).to(device)

    # lm_head legato a embed_tokens (tie_word_embeddings)
    if stages.head and tie:
        model.tie_weights()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
