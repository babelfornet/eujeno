import torch


def sample_token(topk_ids, topk_logits, generated_ids, temperature, top_p,
                 repetition_penalty, generator) -> int:
    """Sceglie il prossimo token dai candidati top-k del nodo head.
    temperature<=0 -> greedy (argmax). Altrimenti: repetition penalty, temperature,
    nucleo top_p, campionamento multinomiale (deterministico se `generator` ha un seed)."""
    logits = torch.tensor(topk_logits, dtype=torch.float32)
    ids = list(topk_ids)
    if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
        gen = set(generated_ids)
        for i, tid in enumerate(ids):
            if tid in gen:
                logits[i] = logits[i] / repetition_penalty if logits[i] > 0 else logits[i] * repetition_penalty
    if temperature is None or temperature <= 0:
        return ids[int(torch.argmax(logits))]
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) <= top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    choice = int(torch.multinomial(sp, 1, generator=generator).item())
    return ids[int(si[choice])]
