#!/usr/bin/env bash
set -euo pipefail
# Execute on the SSH server host. Docker tests/training still run inside the container.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_name="${DATARSI_CONTAINER:-datarsi-dev}"
image="${DATARSI_IMAGE:-datarsi-env:0.3}"
cache_dir="${DATARSI_CACHE:-/home/zqj/.cache/datarsi}"
gpu_devices="${1:-}"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container already exists. Stop/remove this project's container explicitly before recreating." >&2
  exit 1
fi
mkdir -p "$cache_dir"
options=(--name "$container_name" --pull never --init --restart unless-stopped --shm-size 16g
  --mount "type=bind,source=$project_dir,target=/workspace/datarsi"
  --mount "type=bind,source=$cache_dir,target=/workspace/cache"
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock
  --workdir /workspace/datarsi -e HF_HOME=/workspace/cache/huggingface
  -e WANDB_API_KEY --entrypoint /bin/bash)
if [[ -n "$gpu_devices" ]]; then
  # Host indices/UUIDs here; resources.gpu_ids in YAML uses container-local indices.
  options+=(--gpus "\"device=$gpu_devices\"")
else
  options+=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=-1)
fi
docker run -d "${options[@]}" "$image" -lc 'sleep infinity'
