# Real inference observability / 真实推理观测

## What is measured

| Source | Scope | Supported output |
| --- | --- | --- |
| HTTP SSE | This conversation | Response bytes, arrival times, engine-reported token usage |
| NVIDIA `nvidia-smi` | Entire sampled host | Each GPU's utilization, used memory, power in saved samples |
| Apple `ioreg` | Entire driver | GPU utilization and driver in-use unified memory when exposed |
| Linux `/proc/net/dev` | Entire host/interface | Cumulative RX/TX bytes, not NCCL throughput |
| Engine `/metrics` | All engine traffic | Supported vLLM/SGLang KV/token-pool and queue series; labels preserved |
| Chrome/PyTorch trace | Independent capture | Operator/kernel durations, lanes and recorded input shapes |

Unavailable counters stay unknown. Apple IOAccelerator properties are undocumented and may change with macOS; no sudo or privileged helper is installed. Unified-memory driver use is not model allocation and is not added to system RAM. Engine gauges must not be summed across ranks without understanding replication. Metrics fetches use loopback, no HTTP proxy and no redirects.

## 1. Local and remote hardware

Select a deployed service in Playground. By default the collector samples the computer running Workbench. For a cloud engine, enter its existing SSH config alias in **Telemetry SSH alias**, then save. Future conversations use a fixed read-only Python collector through `ssh` with `BatchMode=yes` and `StrictHostKeyChecking=yes`. No password storage, automatic host-key trust, arbitrary command input or sudo.

The alias must point to the actual engine host; Workbench cannot prove that an SSH tunnel and an alias refer to the same machine. Clear the alias and save to return to local sampling. Each response sample carries hostname, scope and timestamp. Linux remote hosts need Python 3; NVIDIA sampling also needs `nvidia-smi`. A failed SSH probe is displayed as unavailable, never replaced silently by local samples. Sampling cadence is approximately collection time plus 2 seconds, not a precise 2 Hz clock. Four chats can produce four collectors; this is not yet a cluster monitoring service.

## 2. Live KV and queue metrics

The collector reads `/metrics` at the selected service URL. vLLM exposes production metrics; SGLang needs `--enable-metrics` (now included in Workbench-generated SGLang deployment commands). Existing deployments must be restarted with that flag if absent. HTTP forwarding for the engine also forwards its metrics endpoint; no additional metrics credential is supported.

The UI shows original series names and labels, including `vllm:kv_cache_usage_perc`, the legacy `vllm:gpu_cache_usage_perc`, `sglang:token_usage` and supported request-queue gauges. Values such as 0.28 are ratios, not 28 bytes. Missing metrics are not zero. The live sample is saved in the same conversation JSONL as the streamed response.

## 3. Real operator/matrix traces

An HTTP chat endpoint does not expose matrix values or kernel execution. Use a short, isolated profiling run:

1. Configure a profiler-capable engine with an output directory on **that engine's host**. Check the installed engine version's documentation and `--help`; profiler flags vary by version.
2. In Playground choose the correct profiler backend, then **Start engine profiling**. The confirmation warns that this changes the entire engine, not only this conversation. Workbench POSTs only `/start_profile` or `/stop_profile` on the registered loopback endpoint. SGLang starts a 10-step CPU/GPU capture; vLLM is manually stopped.
3. Send a short prompt, then **Stop / flush profiling**. Do not leave vLLM profiling running. If a control request times out, state is uncertain: check the engine terminal and explicitly stop it. Browser close does not stop profiling.
4. Find the resulting `.json` or `.json.gz` trace. For a remote host, copy it yourself into a local authorized directory, or open Workbench on that host. Workbench does not silently copy arbitrary remote files.
5. Enter its path and click **Import profiler capture**. Input is read-only; parsed events are saved as a separate replay recording. Drag the time cursor to inspect real operator names, CPU/GPU lanes, durations and input shapes.

For a vLLM **0.12.x** environment, its versioned guide documents `VLLM_TORCH_PROFILER_DIR` and `VLLM_TORCH_PROFILER_RECORD_SHAPES=1`. Newer releases expose profiler configuration differently; do not assume the old variables still work. SGLang documents `SGLANG_TORCH_PROFILER_DIR` and the start/stop endpoints. Shape metadata is optional: a kernel name without recorded shapes cannot reconstruct matrix dimensions reliably.

Import supports Chrome complete-duration (`ph=X`) events, up to 32 MiB compressed and expanded. It retains the first 3,000 valid events, shows 8 lanes and the last 80 rows before the cursor; counts disclose truncation. CPU operators and GPU kernels overlap and must not be summed into wall time. This viewer is not a full Perfetto replacement. It imports neither `.nsys-rep` nor Metal `.gputrace` directly. MLX does not implement these PyTorch profiler endpoints.

## Logs and replay

Conversation streams, host samples and engine series are stored in `.workbench/chat-recordings/<id>.jsonl` (or the configured state directory). Imported traces get their own recording ID and SHA-256; they are explicitly not auto-correlated to a chat. The replay selector works after restart and does not run inference. `operations.log` also records API actions and profiler controls. Logs may contain prompts and responses: protect them and manage retention. Imported trace data is descriptive JSON, never executed.

## Validation boundary

Unit tests cover driver parsing, metric series, SSH argument restrictions, trace import, scope/path checks and persisted replay. Local Apple driver sampling can be tested without a model. Cloud GPU/engine integration must still be validated on the actual deployment; protocol fixtures are not performance benchmarks. True live per-request matrix values and automatic cross-process request/kernel correlation are not implemented.

## References

- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [vLLM profiling](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [Versioned vLLM 0.12 profiling](https://docs.vllm.ai/en/v0.12.0/contributing/profiling/)
- [SGLang production metrics](https://docs.sglang.io/docs/references/production_metrics)
- [SGLang benchmark and profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)
