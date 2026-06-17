import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32   # fp32 su CPU per determinismo (vedi ADR-0001 Fork D)
DEVICE = "cpu"
