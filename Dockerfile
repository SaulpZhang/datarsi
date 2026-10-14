FROM vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
ARG OPENCODE_VERSION=1.19.0
ARG OPENCODE_SHA256=6334d70db3990721ea454357e9cf2573c8685524ad7769af9d52cbffb7008438
RUN mkdir -p /opt/opencode/bin && curl -fsSL --retry 3 "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64-baseline.tar.gz" -o /tmp/opencode.tar.gz && echo "${OPENCODE_SHA256}  /tmp/opencode.tar.gz" | sha256sum -c - && tar -xzf /tmp/opencode.tar.gz -C /opt/opencode/bin && rm /tmp/opencode.tar.gz && /opt/opencode/bin/opencode --version
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
ENV PATH="/opt/opencode/bin:/opt/datarsi-venv/bin:${PATH}"
ENV VIRTUAL_ENV=/opt/datarsi-venv
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "sleep infinity"]
