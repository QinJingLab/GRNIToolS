# GRNIToolS image: CUDA 12.1 + torch 2.1.2, the configuration the published results used.
# Datasets, model weights and code are baked in, so `docker run` needs no extra setup.

# The three network fetches below can be redirected for a slow or restricted network:
#   docker build --build-arg CUDA_IMAGE=... --build-arg MINIFORGE_URL=... --build-arg PIP_INDEX=...
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
ARG PIP_INDEX=https://pypi.org/simple
ARG PIP_WHEEL_INDEX=https://download.pytorch.org/whl/cu121

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CONDA_CHANNEL_PRIORITY=strict \
    PATH=/opt/conda/envs/grnitools/bin:/opt/conda/bin:$PATH \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 git make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Miniforge: conda-forge only, matching the `nodefaults` line in grnitools.yaml.
RUN curl -fsSL -o /tmp/miniforge.sh "$MINIFORGE_URL" && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm -f /tmp/miniforge.sh && conda clean -afy

WORKDIR /opt/GRNIToolS

# Layer 1: the conda environment. Only grnitools.yaml is copied first, so editing code does
# not trigger a re-solve of 242 packages.
COPY grnitools.yaml /tmp/grnitools.yaml
RUN conda env create -f /tmp/grnitools.yaml && conda clean -afy

# Layer 2: the R packages conda lacks, then GRNIToolS itself.
COPY install_deps.R /tmp/install_deps.R
COPY GRNIToolS/GRNIToolS /opt/GRNIToolS/GRNIToolS/GRNIToolS
RUN . "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate grnitools && \
    GRN_PKG=/opt/GRNIToolS/GRNIToolS/GRNIToolS Rscript /tmp/install_deps.R

# Layer 3: the Python libraries of both packages. The requirements reach the CUDA wheels
# through --extra-index-url, whose per-package pages point back at files.pythonhosted.org;
# turning that line into --find-links keeps the same wheels while letting every package
# resolve from a single index.
COPY GRNIToolS_DL/requirements.txt /tmp/dl-requirements.txt
COPY GRNIToolS_scFM/requirements.txt /tmp/scfm-requirements.txt
RUN sed -i "s#^--extra-index-url .*#--find-links ${PIP_WHEEL_INDEX}#" \
        /tmp/dl-requirements.txt /tmp/scfm-requirements.txt
RUN . "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate grnitools && \
    pip install --no-cache-dir --index-url "$PIP_INDEX" -r /tmp/dl-requirements.txt
RUN . "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate grnitools && \
    pip install --no-cache-dir --index-url "$PIP_INDEX" -r /tmp/scfm-requirements.txt

# Layer 4: code, datasets and model weights (~19 GB).
COPY . /opt/GRNIToolS

# The env is active through PATH alone; grn_run.py reads this variable to check it.
ENV CONDA_DEFAULT_ENV=grnitools

CMD ["/bin/bash"]
