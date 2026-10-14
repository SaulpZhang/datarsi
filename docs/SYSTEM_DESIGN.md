# datarsi 系统设计 v0.3

## 目标与边界

用当前 Student 的真实失败指导经验构建，再以执行验证后的经验进行迭代 SFT。固定开发任务用于观察训练收益；测试任务禁止进入修复和训练。当前实现为可运行研究原型，不将“去泄漏”“补证据”作为已经证明的因果机制。

入口为 `python main.py --config config/qwen9.yaml`。配置支持继承、命令行覆盖、阶段选择与断点续跑。默认只在显式指定 GPU 后启动真实训练/推理；CPU 小模型配置用于基础训练测试。

## 模块边界

| 模块 | 职责 | 输出 |
|---|---|---|
| configuration / state | 合并 YAML、验证配置、锁定运行目录、原子保存阶段状态 | resolved-config.yaml、state.json |
| training / data | 精确 loss mask、保留 labels 的 padding、Trainer 全量 SFT | checkpoint、training-metrics.json |
| inference | 托管 vLLM 子进程、检查就绪和模型身份、关闭自己的进程组 | 服务日志、统一模型接口 |
| agent / runtime | OpenCode 后台进程、仓库任务隔离、原生工具调用、执行验证 | 轨迹、patch、verification |
| experience | 失败分类、Student-aware 修复策略、教师补全、重放验证 | 合格轨迹、下一轮训练数据 |
| tracking | W&B 与本地指标、配置及数据 manifest | 每轮指标和数据 lineage |

## 数据流

```text
初始轨迹 -> prepare -> SFT M_t -> vLLM(M_t) -> train-task rollout
                                              -> dev-task evaluation
失败轨迹 -> failure analysis -> teacher-guided continuation -> test verification
验证成功轨迹 + 回放数据 -> prepare -> 下一轮 SFT M_(t+1)
```

模型服务与训练交替占用 GPU，不并发放置所有模型。教师可与 Student 共用模型权重、独立使用私有答案上下文，也可配置不同教师 checkpoint。教师私有 gold patch 不写入公开训练 messages；公开动作的工具观测必须来自真实执行。修复时优先恢复 Student 实际 patch；旧格式轨迹可重放 Bash 前缀。每次模型调用的历史 assistant 输出设为 supervise=false，不重复监督失败或已生成动作。仅有最终测试通过不等于完全消除了间接答案依赖。

## 数据契约

任务 JSONL 必须包含 task_id、split、problem、image、repo_dir、verify_command；可选 base_commit、gold_patch、test_patch。任务镜像不包含宿主机项目挂载，不开放 GPU。Agent 只接收 problem 和实际工具观测；gold_patch 仅在训练集修复时提供给教师。test_patch 仅在 Agent 完成后应用用于验证。

轨迹保存 OpenCode JSON events、model_calls（真实请求 messages、tools 与响应）、actions、observations、patch、模型版本、轮次、usage 和验证状态。任务环境异常与解决失败分开统计。跨 split 的重复 task_id 被拒绝。

SFT 输入保留原始 messages，并按 chat template 与 tokenizer offsets 定位 assistant 内容；系统、用户、tool 和 think 不计算 loss。无法可靠定位时拒绝数据，不退回全 token loss。预分词数据 labels 使用 -100，padding 不覆盖原有 mask，超长轨迹不静默截断。

## 运行与恢复

每个阶段在独立子进程/服务执行，只有产物完成并校验后才写 completed。运行目录保存配置指纹，恢复时配置变化被拒绝。服务关闭后再进入训练；状态文件用于恢复已完成阶段，不声称能从任意 GPU kernel 状态恢复。

修复策略第一版按失败类型选择，并记录各策略的执行成功率与训练后的开发集变化。这是可比较的启发式控制器；后续再加入基于训练收益的策略优化，不将接受率当作真实训练收益。

## 评测与跟踪

W&B 记录训练 loss、轮次、任务通过率、环境错误、token 和时间成本、修复接受率。在线模式需要 WANDB_API_KEY/既有登录；离线模式可后续 wandb sync。大模型权重和原始任务内容默认不上传，数据与 checkpoint 使用本地 manifest 关联。

通用 verify_command 是任务适配器的验证结果，不自动称为 SWE-bench 官方 resolve rate。最终论文结果应增加并使用官方 harness。当前先建立归一化任务格式与 Docker runtime，以避免将数据集的环境适配与方法逻辑耦合。

## 验证顺序

1. CPU：配置覆盖、恢复、mask、轨迹、调度和异常清理。
2. CPU：微型模型全量 SFT、checkpoint 保存恢复、W&B 离线记录。
3. 指定两张卡后：Qwen tokenizer、vLLM、双卡分片训练、显存和吞吐。
4. 少量真实仓库任务，再扩大实验与消融。

## OpenCode 生命周期与网络

固定 OpenCode 1.19.0；Python 通过 Docker API detached exec 执行 Bash 后台命令。超时在容器内用 timeout 约束，Python 另外检查截止时间，退出时移除自己创建的任务容器及所有子进程。任务没有宿主机项目、Docker socket 或 GPU挂载；网络共享开发容器命名空间以访问 localhost vLLM，并非网络隔离。配置目录与session 数据存放任务容器 /tmp，结束即清理。

每个任务启动短期本地模型网关，校验随机凭据，仅转发模型请求，保存无密钥的真实推理上下文。SFT 使用请求工具 schema 渲染 chat template，监督本轮 assistant 文本和完整 tool_call 区域；非文本内容及不可靠掩码直接报错。OpenCode 的工具协议取代原先单个 JSON command/done 协议。
