# Cloud workspaces and environments / 云端工作区与环境

## Execution model

The local UI bootstraps a separate Workbench on a cloud host over an existing SSH alias, then opens a **loopback-only SSH tunnel**. Open that workspace to run all existing model download, scan, resource planning, training, deployment and evaluation features **on the cloud host**. Model weights do not travel through the local computer. Only Workbench source, control requests and UI/log traffic cross SSH.

This is an experimental single-owner workflow, not a multi-tenant cluster scheduler. The system does not purchase cloud resources, edit SSH keys, install GPU drivers or use sudo. Remote machines need working key-based SSH, known_hosts and Python 3.9+. Windows environment provisioning uses WSL/Linux; native Windows is not implemented.

## Follow this path

1. Rent a Linux GPU machine with vendor drivers already working. In a local terminal, configure an alias in `~/.ssh/config` and verify `ssh gpu-server` yourself once. Do not paste a private key or password into Workbench.
2. Open **Hosts & environments / 主机与环境**. Enter the alias and a dedicated remote path such as `/home/ubuntu/model-workbench`. Use unused local and remote ports, e.g. 8876. They may differ.
3. Use the read-only SSH probe to inspect the host. Preview the cloud connection plan and confirm execution. On macOS this opens a real Terminal window; on a headless host it uses the managed terminal log.
4. Wait for the connection task to be running, then click **Open cloud workspace** to reveal its authenticated local link and follow it. If bootstrap is still running, inspect its live log and retry. The current hostname is displayed in the new workspace's **Hosts & environments** page. Keep the tunnel task alive while using it.
5. In the **cloud** workspace, open Cloud model catalog, select HF/HF Mirror/ModelScope, test speed, choose a cloud destination under the authorized remote directory, and download. Existing downloader progress, resume and record management run there. Download SDK installation is optional; Workbench's own downloader uses its built-in HTTP implementation.
6. In the cloud **Hosts & environments** page, select vLLM, SGLang or ms-swift/PyTorch. Supply a NEW environment path (e.g. `/home/ubuntu/model-workbench/env-vllm`) and an installed Python executable. The parent must already exist and be authorized. Preview and confirm.
7. After successful installation, click **Use this Python**. The interpreter path is filled into training, compression and deployment forms for this browser session. Scan the downloaded model, generate the appropriate task plan and execute it. GPU/model compatibility is still checked separately; package installation does not certify it.
8. Deploy a model inside that cloud workspace, check its registered service and open Playground. The cloud backend talks to the model on its own loopback interface, so a second model-port tunnel is unnecessary. Resource samples also come from that host.

For your Mac, use an isolated **MLX LM** environment where appropriate. The standard vLLM/SGLang wheels are not automatically converted to Metal builds. For the GTX 1070 or newer cloud GPUs, library wheel, driver and GPU architecture compatibility still determines support; no fake success is reported when a wheel is missing.

## What environment setup does

| Phase | Actual operation | Evidence |
| --- | --- | --- |
| Create | `python -m venv NEW_PATH` | Command and stdout/stderr |
| Bootstrap pip | Upgrade pip inside that venv | Version/install log |
| Resolve | `pip install --dry-run --report ... --only-binary=:all:` | `resolve.json`, including resolved distribution metadata |
| Lock | Extract exact resolved name/version pairs | `resolved-requirements.txt` |
| Install | Install the resolved versions from PyPI | Installer responses and exit status |
| Check | `pip check`, `pip freeze`, import/GPU smoke test | Terminal log and `environment-result.json` |

The optional version field pins the primary package. If blank, resolution happens at execution time and the chosen versions are recorded, not silently described as a pre-certified combination. This is a version lock, not an immutable wheelhouse or a cryptographic hash-enforced installation. CUDA-specific extra wheel indexes and source builds are not automated in this first recipe set. Wheel-only failure is preferable to silently compiling arbitrary native dependencies for hours.

Provisioning ignores proxy environment variables and pip config, uses official PyPI and disables implicit HTTP proxy discovery. OS-level VPN/TUN routing remains outside this guarantee. No existing environment is overwritten. Failed partial environments remain for inspection; choose a new path for a fresh attempt. No rollback or automatic cleanup deletes your files.

## Command and response trail

- Each plan: `<state>/<job-id>/manifest.json`.
- Live stdout/stderr and every printed command: `<state>/<job-id>/terminal.log`.
- API and execution audit: `<state>/operations.log`.
- Environment command start/exit records: `<state>/<job-id>/environment-commands.jsonl`.
- Resolver result, exact versions and smoke result: same job directory.
- Cloud bootstrap source executed remotely: local `<state>/<job-id>/bootstrap.py`.
- Remote launcher command history: `<remote-root>/bootstrap-commands.jsonl`; startup output: `<remote-root>/server.log`.
- Actual cloud training/deployment/download logs: `<remote-root>/state/`, not local state.

Access tokens are held in permission-restricted access files and are redacted from API audit results. Treat authenticated links and server startup logs as secrets; do not publish them with benchmark logs. Logs preserve commands/responses except for deliberate credential redaction. Existing terminal log viewing returns a bounded tail; the full file remains on disk.

## Disconnect, reconnect and stop

Stopping the **cloud connection task** stops its local SSH tunnel only. The remote workspace and its jobs continue running. Recreate a connection plan with the same alias/root/remote port to reuse the live server and reconnect. A different local tunnel port is allowed. Remote server reuse does not auto-upgrade its running code. Use a separate remote directory for testing a new release, or perform a deliberate maintenance restart after stopping remote jobs.

To stop training or deployment, use Stop **inside the cloud workspace**. This version does not provide a one-click shutdown of the remote control server itself; stop it deliberately from the cloud shell after stopping its jobs, using its recorded command/PID and verifying the process identity. Do not kill an unverified stale PID. Reboot persistence/service-manager installation and automatic remote log backup are not implemented. A local browser refresh preserves server jobs; a remote control-process crash may require recovery checks.

## Validation

Automated tests cover path/alias injection rejection, confirmation boundaries, non-overwrite behavior, failure propagation and a real isolated loopback bootstrap/reuse cycle. SSH transport, cloud wheel installation and real GPU training still require acceptance on your rented server. Unit fixtures are not evidence that every GPU/backend combination works.

Official compatibility references: [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [SGLang installation](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx).
# Discover existing Python environments before installation

Environment creation now defaults to **Conda (Miniforge / Miniconda)**. Existing
installations are detected; users can choose the executable and Python version
(default 3.11) and review the absolute prefix under that installation's `envs`.
The command uses `conda create --prefix ... --override-channels --channel
https://conda.anaconda.org/conda-forge --no-default-packages python=3.11 pip`.
Then the environment's Python installs the model packages using the existing
wheel-resolution/logging workflow. No `conda init`, base modification or automatic
Miniforge installer is run. Missing Conda requires explicit user setup or selecting
the optional venv mode. The target directory must be authorized and not exist.
macOS/Linux are the tested control-plane targets; Windows path handling is included
but has not been validated on a Windows host. GPU recipes still require Linux/WSL;
Conda availability does not remove engine or GPU restrictions.

Training, compression and deployment provide **Find Python environments**. The scan
checks the current interpreter, PATH, active venv/Conda, common environment folders
and an optional trusted folder. It probes up to 24 interpreters with isolated Python
metadata commands; it does not install packages, import models or scan every disk
recursively. Only scan interpreters you trust: probing executes their Python binary.

Select one of the package-compatible candidates explicitly. Selection updates only
the current task and invalidates its old execution plan. Package presence is not
GPU, driver, architecture or model validation; run deployment/training preflight next.
Raw reports include paths, versions, probe commands and responses in the operation audit.

In **Hosts & environments**, enter a configured SSH alias and scan its environments.
Remote results are labelled and cannot populate a local task. Open that host's cloud
workspace, then scan/select there. If nothing matches, review the fresh-venv command
plan, choose its location and confirm execution. Existing environments are never
overwritten; no install runs merely because you scanned. Installation commands and
responses remain in the task logs. GPU recipes currently target Linux, MLX Apple
Silicon; automatic CUDA driver installation and universal compatibility are not offered.
