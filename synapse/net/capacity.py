"""Stima capacità di un nodo: quanti layer regge data la RAM, e probe risorse."""
import os

_GB = 1024 ** 3


def fit_layers(dims: dict, bytes_per_param: int, ram_gb: float, reserve: float = 0.2) -> dict:
    """Dato il modello (dims), la dimensione in byte di un parametro e la RAM
    disponibile in GB, stima quanti layer decoder regge il nodo."""
    hidden = dims["hidden_size"]
    nl = dims["num_layers"]
    heads = dims["num_attention_heads"]
    kv = dims.get("num_key_value_heads") or heads
    inter = dims.get("intermediate_size") or (4 * hidden)
    vocab = dims.get("vocab_size") or 0
    kv_dim = hidden * kv / heads
    params_layer = 2 * hidden ** 2 + 2 * hidden * kv_dim + 3 * hidden * inter
    ram_layer = params_layer * bytes_per_param
    ram_embed_head = vocab * hidden * bytes_per_param
    usable = ram_gb * (1 - reserve) * _GB
    max_layers = max(0, int(usable // ram_layer)) if ram_layer > 0 else 0
    return {
        "ram_per_layer_gb": round(ram_layer / _GB, 3),
        "ram_embed_head_gb": round(ram_embed_head / _GB, 3),
        "max_decoder_layers": min(max_layers, nl),
        "fits_whole_model": (nl * ram_layer + ram_embed_head) <= usable,
    }


def probe_capacity() -> dict:
    """RAM totale/libera (GB) e numero di CPU. Usa psutil se presente, altrimenti stdlib."""
    cpu = os.cpu_count() or 1
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"ram_total_gb": round(vm.total / _GB, 2),
                "ram_free_gb": round(vm.available / _GB, 2), "cpu_count": cpu}
    except Exception:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            return {"ram_total_gb": round(total / _GB, 2),
                    "ram_free_gb": round(free / _GB, 2), "cpu_count": cpu}
        except (ValueError, OSError, AttributeError):
            return {"ram_total_gb": None, "ram_free_gb": None, "cpu_count": cpu}
