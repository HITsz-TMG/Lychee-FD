ARG BASE_IMAGE=lychee-fd-early-exit-backend:dev

FROM ${BASE_IMAGE}

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json /app/frontend/

RUN cd /app/frontend \
  && npm ci

COPY . /app

RUN chmod +x /app/scripts/*.sh

ENV STEPAUDIO_CONDA_ENV_PATH=/opt/conda/envs/fdmodel-demo
ENV STEPAUDIO_VLLM_SOURCE_DIR=/app/third_party/vllm
ENV STEPAUDIO_VLLM_SYNC_FLASH_ATTN=1
ENV STEPAUDIO_VLLM_FORCE_SYNC_FLASH_ATTN=1
ENV STEPAUDIO2_SOURCE_DIR=/app/third_party/Step-Audio2
ENV PYTHONPATH=/app:/app/third_party/Step-Audio2
ENV PATH=/opt/conda/envs/fdmodel-demo/bin:/opt/conda/bin:${PATH}
ENV PYTHONUNBUFFERED=1

EXPOSE 7860 8084 8091

ENTRYPOINT ["/app/scripts/start_frontend_dev.sh"]
CMD ["prod", "public"]
