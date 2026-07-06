# Lychee-FD

Lychee-FD full-duplex realtime speech demo.

This repository does not include model weights. Use the prebuilt Docker image for the fastest setup, then mount local model weights into the container.

## Quick Start With Docker

Prepare configuration:

```bash
cp .env.docker.example .env
```

Edit `.env`:

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/your-github-username/lychee-fd:latest
HOST_MODEL_ROOT=/path/to/model-root
STEPAUDIO_MODEL_PATH=/data/zxy/model/path/to/lychee-fd/checkpoint
STEPAUDIO_T2W_MODEL_PATH=/data/zxy/model/token2wav
```

The host model root is mounted into the container as `/data/zxy/model`.

Start:

```bash
docker compose pull
docker compose up
```

Open:

```text
http://127.0.0.1:8084
```

## Build From Source

The default `compose.yaml` uses a prebuilt image. To rebuild the app image locally:

```bash
docker compose -f compose.yaml -f compose.build.yaml build frontend
```

The local build expects the base runtime image in `.env`:

```dotenv
LYCHEE_FD_BASE_IMAGE=lychee-fd-early-exit-backend:dev
```

## Publish Docker Image

See [DOCKER_PUBLISH.md](DOCKER_PUBLISH.md).
