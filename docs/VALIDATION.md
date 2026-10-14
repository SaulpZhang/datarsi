# Validation — 2026-10-15

## Passed in the SSH server Docker container

- Rebuilt pinned Dockerfile as `datarsi-env:0.2`; recreated `datarsi-dev` from that image.
- Dependency integrity: `python -m pip check` reported no broken requirements.
- Full opt-in suite: **25 passed** with Docker and real-tokenizer checks enabled.
- Config YAML inheritance, dotted overrides, malformed config rejection and two-GPU limit.
- State failure/retry, completed-stage reuse, artifact absence, config/data fingerprints and exclusive run lock.
- Assistant loss mask excludes user/tool observations, think and failed-prefix actions; collator preserves labels and masks padding.
- Actual `Qwen/Qwen3.5-9B` tokenizer: environment and private reasoning excluded, failed prefix context-only; 6 supervised tokens out of 59 in the regression example. Tokenizer files only, no language-model weights loaded.
- Tiny random GPT-2 CPU full SFT: two steps, finite loss, checkpoint save and reload.
- `main.py --config cpu_smoke.yaml --stage train --set experiment.name=cpu-smoke-verified`, followed by `--resume`, completed. Resume reused completed stages.
- W&B offline pipeline and training records generated under `runs/cpu-smoke-verified/wandb/`.
- Real Docker task fixture: initialize repository, model-interface action edits a file, execute verification, extract patch and remove task container. The action provider is a fixture, not a real LLM.
- Simulated service readiness verifies served-model identity and cleans up only owned process groups; timeout cleanup tested.
- Failed task infrastructure is retried on resume and does not silently produce an all-invalid evaluation.

Command:

```bash
DATARSI_DOCKER_TESTS=1 DATARSI_TOKENIZER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Not yet validated

- Real Qwen weight loading, GPU vLLM startup and inference.
- Dual-H200 full SFT, FSDP2 wrapping compatibility, peak VRAM, throughput and GPU checkpoint resume.
- SWE-Gym task-image adapters and official SWE-bench harness metrics.
- Scientific performance gains, multi-round strategy evolution or novelty claims.
- W&B online sync: no API credential was supplied; offline tracking is operational.

The container has no GPU device requests. Two SWIG deprecation warnings were emitted by dependencies; tests passed. Raw data, model weights and W&B files are excluded from Git.
