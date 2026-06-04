# Rinha de Backend 2026 — Zig

Fraud-detection vector-search API in **Zig**, built for the lowest possible p99
with **provably exact** k-NN classification (zero false positives / negatives).

For each `POST /fraud-score` the service turns the transaction into a 14-dim
vector, finds the 5 nearest of 3,000,000 reference vectors (squared Euclidean),
and returns `approved = (frauds_among_5 < 3)`.

## Key ideas

- **Exact integer search.** Every reference & query coordinate the generator
  emits is rounded to 4 decimals, so multiplying by `10000` yields an exact
  `i16`. Squared-Euclidean distance in integers is therefore *bit-identical* to
  the float64 brute force the grader uses to label payloads — int16 is lossless,
  not an approximation. This is what guarantees `E = 0` (no FP/FN). Verified by
  differential testing against the official data-generator on 130k+ payloads
  (1500+ edge cases at the `score == 0.6` tie).

- **kd-tree with incremental branch-and-bound.** A median-split kd-tree
  (leaves of 64 vectors stored as SoA blocks of 8) is searched with an exact
  per-dimension lower bound; the far subtree is visited only when it could still
  contain a closer neighbour. This returns the *true* 5 nearest while touching a
  tiny fraction of the 3M vectors. Distances are computed 8 lanes at a time with
  AVX2 (`@Vector`), and a block-level SIMD min-skip avoids touching the heap for
  blocks that cannot beat the current 5th-best.

- **No per-request allocation or serialization.** The only possible response
  bodies are the 6 fraud-count outcomes, precomputed in full (headers + body) at
  comptime. The index is a prebuilt binary blob, `mmap`'d (`MAP_POPULATE`) at
  startup — no JSON parsing at runtime.

- **SCM_RIGHTS load balancer.** The load balancer accepts TCP connections on
  `:9999`, sets `TCP_NODELAY`, and hands each accepted socket to an API worker
  (round-robin) over a Unix `SOCK_SEQPACKET` channel using `SCM_RIGHTS` — it
  never reads the HTTP payload. Each API owns its connections end-to-end via a
  single-threaded `epoll` loop with HTTP/1.1 keep-alive.

## Architecture

```
client --TCP:9999--> [ lb ] --SCM_RIGHTS(fd)--> [ api1 ] (epoll, mmap index)
                        |   round-robin    \---> [ api2 ] (epoll, mmap index)
```

- `lb`: bridge network, publishes `:9999`. 0.10 CPU / 30 MB.
- `api1`, `api2`: `network_mode: none`, reachable only via the shared tmpfs
  Unix socket. 0.45 CPU / 160 MB each. Total: **1.0 CPU / 350 MB**.

Static `musl` binaries (`-mcpu=haswell`, AVX2) on a `scratch` image.

## Layout

```
src/
  vectorize.zig   payload JSON -> 14-dim int16 vector (exact, comptime mcc table)
  refs.zig        references.json -> int16 reference vectors
  index.zig       kd-tree build + exact branch-and-bound search + (de)serialize
  oracle.zig      brute-force int16 k-NN (the correctness oracle)
  http.zig        precomputed responses + request framing
  api.zig         epoll worker: SCM_RIGHTS fd intake, keep-alive HTTP
  lb.zig          SCM_RIGHTS fd-passing load balancer
  indexer.zig     build-time: references.json -> index.bin
build.zig         3 static binaries (lb, api, indexer), musl + haswell
Dockerfile        reproducible from-source build (zig + refs -> scratch image)
```

## Build & run

```sh
zig build                 # -> zig-out/bin/{lb,api,indexer}
zig build test            # unit tests
docker build -t rinha-zig .
docker compose up         # see docker-compose.yml
```

License: MIT.
