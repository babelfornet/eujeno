# Nodo worker Eujeno — carica in RAM SOLO i layer assegnati (partial loading),
# così gira su container con poca memoria. Si connette in uscita al coordinator.
FROM python:3.12-slim

WORKDIR /app

# torch CPU-only in un layer SEPARATO (cache stabile finché non cambi versione):
# su Linux il torch di default tira dentro GB di wheel CUDA inutili per la CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# il codice cambia spesso: questo layer si ricostruisce, ma torch resta in cache
COPY pyproject.toml ./
COPY eujeno ./eujeno
RUN pip install --no-cache-dir . && rm -rf /root/.cache/pip

# cache modelli condivisa via volume (vedi compose): evita ri-download
ENV HF_HOME=/hf

ENTRYPOINT ["eujeno"]
