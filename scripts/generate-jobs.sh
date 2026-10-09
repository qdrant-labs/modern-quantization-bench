#!/usr/bin/env bash
#
# Writes one kubench job file per dataset into ./jobs. It applies nothing.
# Apply them by hand, one job per lane at a time:
#
#   kubectl apply -f jobs/a-01-3-small-512.yaml -f jobs/b-01-3-small-1024.yaml
#
# Lane a runs on clelias-dell-pro, lane b on laptop-lino2gmk. Each lane has its own Qdrant
# cluster, so a job of lane a and a job of lane b can run together. Never run two jobs of
# the same lane together, because they would share a cluster.
#
# Volumes are NFS volumes, so the pod mounts the share itself and never depends on a host mount.
#

set -euo pipefail

IMAGE=${IMAGE:-100.64.73.86:5000/binquant-bench:latest}
NFS_SERVER=${NFS_SERVER:-100.64.73.86}
SHARED=${SHARED:-/srv/kubench-shared/binquant}
OUT_DIR=${OUT_DIR:-jobs}

LANE_NAMES=(a b)
LANE_NODES=(clelias-dell-pro laptop-lino2gmk)
LANE_SECRETS=(qdrant-creds qdrant-creds-1)

# ada002-1536 is done and 3-large-3072 is running, so both are left out.
DATASETS=(
  3-small-512 3-small-1024 3-small-1536 3-large-1024 3-large-1536
  arxiv-titles-instructorxl arxiv-abstracts-instructorxl bge-m3-ads gte-ads gte-product-ads
  wolt-food-clip fineweb-10b coyo-ve
)

mkdir -p "$OUT_DIR"
for i in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$i]}
  lane=$((i % ${#LANE_NAMES[@]}))
  file=$(printf '%s/%s-%02d-%s.yaml' "$OUT_DIR" "${LANE_NAMES[$lane]}" $((i / ${#LANE_NAMES[@]} + 1)) "$dataset")
  kubench job \
    --image "$IMAGE" \
    --metadata "binquant-$dataset" \
    --container "binquant-$dataset" \
    --node-name "${LANE_NODES[$lane]}" \
    --env-secret "${LANE_SECRETS[$lane]}" \
    --nfs-volume "$NFS_SERVER:$SHARED/cache:/app/.cache" \
    --nfs-volume "$NFS_SERVER:$SHARED/results:/app/results" \
    --backoff 0 \
    --outfile "$file" \
    -- bench run --dataset "$dataset"
done
echo "wrote $(ls "$OUT_DIR" | wc -l) job files in $OUT_DIR"
