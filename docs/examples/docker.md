# 3-node test — 1 host + 2 Docker containers

A small model (Qwen2.5-0.5B, 24 layers) split across **3 nodes**: one on the host (this machine) and two in Docker containers. Each node loads **only its own layers** into RAM (partial loading). It uses the **coordinator** mode (the containers connect outbound to the host).

Distribution of the 24 layers:

| Node | Where | Stage |
|------|------|-------|
| coordinator | host | (routes, does not run layers) |
| host-node | host | `embed,decoder:0-8` |
| node-mid | container | `decoder:8-16` |
| node-tail | container | `decoder:16-24,head` |

## Startup

```bash
# 0) (once) download the model on the host so the containers reuse it from the shared cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# 1) Coordinator on the host (port 9000, reachable by the containers)
axyn coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000 &

# 2) Node on the host: embedding + first 8 layers
axyn serve --coordinator ws://127.0.0.1:9000/node --stages "embed,decoder:0-8" &

# 3) Two nodes in the containers (built automatically the first time)
docker compose -f docker/compose.yaml up --build -d

# 4) Wait for all 3 nodes to register and the coverage to be complete
curl -s http://127.0.0.1:9000/registry        # must list 3 nodes

# 5) Test a prompt from this machine
axyn --json infer --coordinator http://127.0.0.1:9000 --prompt "The capital of Italy is"
```

## Verifying the memory benefit

```bash
docker stats --no-stream    # container RAM: each loads only its own layers, not the whole model
```

## Stop

```bash
docker compose -f docker/compose.yaml down
kill %1 %2     # coordinator + host-node
```

> The containers share the host's Hugging Face cache (`~/.cache/huggingface` mounted at `/hf`), so they don't re-download the model. Each node materializes only its assigned layers in RAM; the rest stay on the `meta` device (zero memory).
