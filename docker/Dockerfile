# Nodo worker Synapse — carica in RAM SOLO i layer assegnati (partial loading),
# così gira su container con poca memoria. Si connette in uscita al coordinator.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY synapse ./synapse

# torch CPU + dipendenze. Su arm64 (Apple Silicon) pip prende già la wheel CPU.
RUN pip install --no-cache-dir . && rm -rf /root/.cache/pip

# cache modelli condivisa via volume (vedi compose): evita ri-download
ENV HF_HOME=/hf

ENTRYPOINT ["synapse"]
