# syntax=docker/dockerfile:1.6

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG RUST_VERSION=1.78.0
ARG CUDA_ARCHES="sm_89"

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

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
    ARCHES="${CUDA_ARCHES}" ./build_cubin_ada.sh && \
    if [ ! -f sha256_kernel.ptx ]; then \
        FIRST_ARCH=$(echo "${CUDA_ARCHES}" | cut -d, -f1); \
        COMPUTE_ARCH=$(echo "${FIRST_ARCH}" | sed 's/sm_/compute_/'); \
        nvcc -O3 -ptx -arch=${COMPUTE_ARCH} sha256_kernel.cu -o sha256_kernel.ptx; \
    fi

RUN cargo build --release --locked

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i '/jammy-backports/d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/target/release/bro /usr/local/bin/bro
COPY --from=build /app/sha256_kernel.* ./
RUN rm -f sha256_kernel.cu

EXPOSE 8001

ENTRYPOINT ["/usr/local/bin/bro"]
