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
