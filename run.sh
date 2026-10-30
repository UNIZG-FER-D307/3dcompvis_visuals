#!/usr/bin/env bash
# Usage: ./run.sh build | serve | test [pytest args] | e2e [pytest args]
set -euo pipefail
cd "$(dirname "$0")"
source vars.env
IMAGE=rs-demo
MOUNT=(-v "$PWD:/app:Z" -w /app)

# Device nodes of attached RealSense cameras (V4L2 video/media nodes + USB node).
# Rootless podman does not pass /dev/bus/usb reliably, so list them explicitly.
realsense_devices() {
    local f
    for f in /sys/class/video4linux/*/name /sys/bus/media/devices/*/model; do
        grep -q RealSense "$f" && echo "--device=/dev/$(basename "$(dirname "$f")")"
    done
    for f in /sys/bus/usb/devices/*/product; do
        grep -q RealSense "$f" || continue
        printf -- "--device=/dev/bus/usb/%03d/%03d\n" \
            "$(cat "${f%/*}/busnum")" "$(cat "${f%/*}/devnum")"
    done
    return 0
}

case "${1:-}" in
build)
    podman build --target base -t "$IMAGE" .
    podman build --target test -t "$IMAGE-test" .
    podman build --target e2e -t "$IMAGE-e2e" .
    ;;
serve)
    mapfile -t DEVICES < <(realsense_devices)
    [ ${#DEVICES[@]} -gt 0 ] || echo "No RealSense camera found; serving a synthetic scene" >&2
    podman run --rm ${DEVICES[@]+"${DEVICES[@]}"} \
        --memory "$MEM_LIMIT" -p "$PORT:8000" "${MOUNT[@]}" "$IMAGE" \
        uvicorn app.server:main --factory --host 0.0.0.0 --port 8000 \
        --reload --reload-dir app
    ;;
test)
    shift
    podman run --rm --memory "$MEM_LIMIT" "${MOUNT[@]}" "$IMAGE-test" \
        python -m pytest -q "$@"
    ;;
e2e)
    shift
    podman run --rm --memory "$MEM_LIMIT" "${MOUNT[@]}" "$IMAGE-e2e" \
        python -m pytest -q tests/e2e "$@"
    ;;
*)
    echo "Usage: $0 build | serve | test [pytest args] | e2e [pytest args]" >&2
    exit 1
    ;;
esac
