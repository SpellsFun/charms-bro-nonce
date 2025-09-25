# syntax=docker/dockerfile:1.6

ARG CUDA_VERSION=12.5.1
ARG UBUNTU_VERSION=22.04
ARG RUST_VERSION=1.78.0
# 支持多架构：sm_89 (RTX 4090), sm_90 (H100), sm_100 (RTX 5090 Blackwell)
# 留空则自动检测
ARG CUDA_ARCHES=""

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG RUST_VERSION
ARG CUDA_ARCHES
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH="/usr/local/cargo/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive

RUN sed -i '/jammy-backports/d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        pkg-config \
        git \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION}

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY build_cubin_ada.sh sha256_kernel.cu ./

RUN chmod +x build_cubin_ada.sh && \
    if [ -n "${CUDA_ARCHES}" ]; then \
        echo "[Docker] Building for specified architectures: ${CUDA_ARCHES}"; \
        ARCHES="${CUDA_ARCHES}" ./build_cubin_ada.sh; \
    else \
        echo "[Docker] Auto-detecting GPU architectures"; \
        ./build_cubin_ada.sh; \
    fi && \
    # 确保生成 PTX 作为通用后备
    if [ ! -f sha256_kernel.ptx ]; then \
        echo "[Docker] Generating PTX for forward compatibility"; \
        nvcc -O3 -ptx -arch=compute_90 sha256_kernel.cu -o sha256_kernel.ptx || \
        nvcc -O3 -ptx -arch=compute_89 sha256_kernel.cu -o sha256_kernel.ptx || \
        nvcc -O3 -ptx -arch=compute_80 sha256_kernel.cu -o sha256_kernel.ptx; \
    fi

RUN cargo build --release --locked

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i '/jammy-backports/d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/bro /usr/local/bin/bro
COPY --from=builder /app/sha256_kernel.* ./
RUN rm -f sha256_kernel.cu

EXPOSE 8001

ENTRYPOINT ["/usr/local/bin/bro"]
