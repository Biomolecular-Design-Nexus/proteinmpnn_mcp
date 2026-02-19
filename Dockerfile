FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    loguru \
    click \
    pandas \
    tqdm \
    && pip install --no-cache-dir --ignore-installed fastmcp \
    && pip install --no-cache-dir -U cryptography certifi

# Clone ProteinMPNN repository (includes model weights ~71MB)
RUN git clone https://github.com/dauparas/ProteinMPNN.git /app/repo/ProteinMPNN

# Copy model weights into examples/data/ so get_model_path() finds them
RUN mkdir -p /app/examples/data && \
    cp -r /app/repo/ProteinMPNN/vanilla_model_weights /app/examples/data/ && \
    cp -r /app/repo/ProteinMPNN/ca_model_weights /app/examples/data/ && \
    cp -r /app/repo/ProteinMPNN/soluble_model_weights /app/examples/data/

# Pre-download / verify model weights are loadable (saves first-run latency)
RUN python -c "\
import torch; \
import glob; \
weights = glob.glob('/app/examples/data/vanilla_model_weights/*.pt'); \
assert len(weights) > 0, 'No vanilla weights found'; \
torch.load(weights[0], map_location='cpu', weights_only=False); \
print(f'Verified {len(weights)} vanilla model weights'); \
weights_ca = glob.glob('/app/examples/data/ca_model_weights/*.pt'); \
print(f'Verified {len(weights_ca)} CA model weights'); \
weights_sol = glob.glob('/app/examples/data/soluble_model_weights/*.pt'); \
print(f'Verified {len(weights_sol)} soluble model weights'); \
print('All checkpoints ready')"

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Copy example input PDB files
COPY examples/ ./examples/

# Create working directories
RUN mkdir -p /app/results /app/jobs /app/tmp

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]
