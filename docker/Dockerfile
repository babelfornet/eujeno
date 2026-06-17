# Nodo worker Synapse — carica in RAM SOLO i layer assegnati (partial loading),
# così gira su container con poca memoria. Si connette in uscita al coordinator.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY synapse ./synapse

# torch CPU-only PRIMA (su Linux il torch di default tira dentro GB di wheel CUDA
# inutili per la CPU); poi il resto delle dipendenze.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip

# cache modelli condivisa via volume (vedi compose): evita ri-download
ENV HF_HOME=/hf

ENTRYPOINT ["synapse"]
