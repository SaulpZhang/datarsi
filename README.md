# datarsi

Student-aware experience learning for coding agents. 主入口是 `main.py`，实验配置放在 `config/`。

## 当前实现

配置继承/覆盖、阶段状态与恢复、Transformers 全量 SFT、自动管理 vLLM、Docker 仓库执行、失败分类与答案辅助修复、W&B 跟踪。修复控制器当前为启发式策略；不是已经验证的策略进化算法。

系统设计与验证边界见 [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)。

## 服务器环境

SSH `H200-zqj`，宿主机 `/home/zqj/datarsi`，容器 `datarsi-dev`，容器目录 `/workspace/datarsi`。所有运行和测试在容器内进行。Python 环境 `/opt/datarsi-venv`。GPU 尚未指定。

```bash
docker exec -it datarsi-dev bash
source /opt/datarsi-venv/bin/activate
cd /workspace/datarsi
```

查看完整流程计划，不下载模型或启动作业：

```bash
python main.py --config config/qwen9.yaml --dry-run
```

只运行一个阶段，或覆盖参数：

```bash
python main.py --config cpu_smoke.yaml --stage train
python main.py --config qwen9.yaml --set experiment.name=qwen9-v2 --set resources.gpu_ids='[0,1]' --dry-run
```

GPU 分配、任务数据和初始轨迹就绪后执行完整流程：

```bash
python main.py --config config/qwen9.yaml --set resources.gpu_ids='[0,1]'
# 中断后继续：配置必须与原运行一致
python main.py --config config/qwen9.yaml --set resources.gpu_ids='[0,1]' --resume
```

训练结束后调度器自动关闭训练进程，启动 checkpoint 的 vLLM，等待模型身份匹配，再启动任务。修复服务结束后关闭 vLLM，再进入下一轮训练。默认完整流程不接受无 GPU 运行。

`--stage` 支持 `prepare/train/rollout/repair/evaluate/pipeline`。CPU smoke 是两步随机微型模型训练，不代表真实 Qwen/vLLM 测试。

## 数据与实验

默认训练输入为 `data/bootstrap.jsonl`（已验证的训练 messages 轨迹），或配置 `data.train_file` 为预分词 JSONL。预分词记录需要 `input_ids/attention_mask/labels`，labels=-100 表示不监督。环境与 think 在输入上下文中保留，但不计算 loss。Mask 无法可靠定位或轨迹过长时直接报错。

任务文件为 `data/tasks/train.jsonl` 和 `data/tasks/dev.jsonl`。每行契约：

```json
{"task_id":"repo-123", "split":"train", "problem":"Issue description", "image":"PREBUILT_TASK_IMAGE", "repo_dir":"/testbed", "verify_command":"pytest ...", "base_commit":"COMMIT", "gold_patch":"diff ...", "test_patch":"diff ..."}
```

`image` 必须对应可执行任务环境，`verify_command` 必须由数据集适配器确定。没有静默下载所有任务镜像；可通过配置开启缺失镜像拉取。Agent 的任务容器不挂载宿主机项目、Docker socket 或 GPU。教师私有提示不进入训练 messages。dev/test 不用于答案辅助修复；bootstrap 与 dev 的 task_id 重叠被拒绝。

这里只报告通用任务验证通过率；SWE-bench 官方结果需要后续接入官方评测 harness。当前未自动认定任何任务测试命令等价于官方 resolve rate。

运行产物在 `runs/<experiment.name>/`：resolved-config.yaml、state.json、metrics.jsonl、每轮 checkpoint、轨迹、修复数据和服务日志。恢复复用完成阶段与已保存任务；配置变化或产物丢失会拒绝恢复。

## W&B

`tracking.mode: offline` 默认保存到运行目录，无需登录。在线跟踪可设 `tracking.mode: online`，通过环境变量 `WANDB_API_KEY` 或容器内 `wandb login` 配置，不把密钥写入 YAML/Git。

离线记录后可使用 `wandb sync <offline-run-directory>`。训练子运行按轮次分组；顶层运行记录任务指标、修复接受率和开发集变化。仅上传小型状态 manifest，模型权重和原始轨迹默认不上传。

## 环境复现与 GPU

Dockerfile 固定基础镜像摘要和直接依赖；`environment/` 保存版本快照。服务器的开发环境镜像为 `datarsi-env:0.2`。

创建脚本 `scripts/container.sh` 默认不分配 GPU；传入 `0,1` 等宿主机卡编号/UUID 可创建限制设备的容器。已有容器应先保留环境镜像和项目产物，再显式重建。YAML 的 gpu_ids 为容器内逻辑编号，与宿主机编号不同。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
# 可选：真实 Docker 与实际 Qwen tokenizer（不加载模型权重）
DATARSI_DOCKER_TESTS=1 DATARSI_TOKENIZER_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

CPU 测试检查配置、loss mask、状态恢复、私有信息隔离和服务清理；GPU 分配后另测 Qwen tokenizer/模型类、vLLM 和双卡 FSDP。当前 FSDP2 YAML 是待 GPU 验证的起点。

本次已验证的范围见 [docs/VALIDATION.md](docs/VALIDATION.md)。
