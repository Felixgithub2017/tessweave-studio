# Contributing

Add functionality as a bounded adapter, with model/data/resource contracts and tests. Never mark a model verified solely because its model_type is recognized. Never accept arbitrary shell strings from model metadata or browser requests. Unknown information must stay unknown, not become zero or a fabricated benchmark.

Run `python -m unittest discover -s tests -v`. GPU PRs should include reproducible environment, immutable model revision, commands, actual metrics, and failure/cleanup evidence. Do not commit weights, credentials, private samples or generated state.

Priority roadmap: real GPU validation of current recipes; exact adapter parameter/activation accounting; service readiness and standardized Prometheus metrics; versioned capability registry; multimodal generation adapters; PPO role orchestration; distributed agents and recovery; low-cost configuration search; pruning plus recovery training. These are not claimed as completed in 0.1.
