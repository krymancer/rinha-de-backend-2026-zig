# syntax=docker/dockerfile:1
# Reproducible from-source build: downloads Zig, builds the static musl binaries,
# downloads the official 3M references, builds the kd-tree index, then assembles a
# scratch image containing only the two binaries + the prebuilt index.

FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG ZIG_VERSION=0.16.0
RUN curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
      | tar -xJ -C /opt \
    && ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

WORKDIR /build
COPY build.zig ./
COPY src ./src
RUN zig build

# Fetch official references (fixed for the edition) and bake the index.
ARG REFS_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
ARG LEAF_MAX=64
RUN curl -fsSL "${REFS_URL}" -o refs.json.gz \
    && gunzip refs.json.gz \
    && ./zig-out/bin/indexer refs.json /build/index.bin ${LEAF_MAX} \
    && rm -f refs.json

FROM scratch
COPY --from=builder /build/zig-out/bin/lb /lb
COPY --from=builder /build/zig-out/bin/api /api
COPY --from=builder /build/index.bin /index.bin
