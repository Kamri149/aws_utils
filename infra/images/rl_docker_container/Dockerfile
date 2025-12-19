# CPU-only, small-ish base. Good default for single-VM notebook + RL.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS deps (add more only if you need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY infra/images/rl_docker_container/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# Copy full repo into image (context should be repo root)
COPY . /app

# Install your project (assumes pyproject.toml at repo root)
# If you want editable installs, you'd need a different approach; keep it standard here.
RUN pip install -e . || pip install .

# Expose Jupyter port
EXPOSE 8888

# Default command: JupyterLab (no token only if you're tunneling to localhost; otherwise keep token)
# Here we keep token enabled by default for safety; you can override at runtime if you prefer.
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.allow_origin='*'"]
