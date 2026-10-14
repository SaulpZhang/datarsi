FROM vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
WORKDIR /workspace/datarsi
RUN apt-get update -qq && apt-get install -y --no-install-recommends pkg-config libcairo2-dev git && rm -rf /var/lib/apt/lists/*
RUN git config --system --add safe.directory /workspace/datarsi
COPY environment/requirements.txt /tmp/datarsi-requirements.txt
RUN python3 -m venv --system-site-packages /opt/datarsi-venv && /opt/datarsi-venv/bin/python -m pip install -r /tmp/datarsi-requirements.txt
COPY pyproject.toml main.py ./
COPY src/ ./src/
COPY config/ ./config/
COPY examples/ ./examples/
RUN /opt/datarsi-venv/bin/python -m pip install --no-deps --no-build-isolation -e .
ENV PATH="/opt/datarsi-venv/bin:${PATH}"
ENV VIRTUAL_ENV=/opt/datarsi-venv
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "sleep infinity"]
