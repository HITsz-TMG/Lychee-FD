ARG BASE_IMAGE=lychee-fd-early-exit-flash-attn:dev

FROM ${BASE_IMAGE}

# The base image must provide the environment only. Clear runtime source dirs
# defensively so old app trees are not visible if a non-clean base is used.
RUN rm -rf /app /workspace \
  && mkdir -p /app

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json /app/frontend/

RUN cd /app/frontend \
  && npm ci

COPY . /app

RUN chmod +x /app/scripts/*.sh

ENV LYCHEEFD_CONDA_ENV_PATH=/opt/conda/envs/fdmodel-demo
ENV LYCHEEFD_VLLM_SOURCE_DIR=/app/third_party/vllm
ENV LYCHEEFD_VLLM_SYNC_FLASH_ATTN=1
ENV LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN=1
ENV STEPAUDIO2_SOURCE_DIR=/app/third_party/Step-Audio2
ENV PYTHONPATH=/app:/app/third_party/Step-Audio2
ENV PATH=/opt/conda/envs/fdmodel-demo/bin:/opt/conda/bin:${PATH}
ENV PYTHONUNBUFFERED=1

EXPOSE 7860 8084 8091

ENTRYPOINT ["/app/scripts/start_frontend_dev.sh"]
CMD ["prod", "public"]
