# Initial public source milestone

Date: 2026-10-08. Display identity: **TessWeave Studio**. Repository and command compatibility: **Model Workbench**.

## Checks performed

- Control-plane regression: **168 tests passed** on local Python 3.10.20. HTTP tests used isolated loopback ports; no paid GPU or remote server was rented.
- Source and wheel builds completed. The wheel contains the packaged SVG icon and both bundled JavaScript libraries' license texts. Private runtime/cache directories were absent from inspected archives.
- Public-source review checked staged files for known credential patterns, personal absolute paths, runtime artifacts and oversized files; no matches remained. Pattern scanning is not a guarantee that all security defects are absent.
- The redistributed Gemma report matched the SHA-256 in its separate attribution notice; local-only report archives are not uploaded.
- Both retained historical documentation screenshots were visually reviewed. Other screenshots and the local-path-bearing MLX run record stay on disk and are excluded from Git. Original model files, conversations and jobs are not deleted.
- The original SVG banner was rendered and visually checked. The server regression checks the icon route and SVG MIME type.

## What this milestone does not establish

It is not a GPU certification, a security audit, an optimality proof for the planner, a completed performance comparison, a PyPI release or an arXiv submission. Source publication does not validate every backend/model pair. Hosted CI results are separate from the local checks above.

The [Engine connection guide](TESSWEAVE.md) describes the existing manual HTTP route and its restrictions. It does not claim a newly validated real-model end-to-end Studio/Engine run or a one-click lifecycle adapter.

## Source publication boundary

`.workbench/`, `.knowledge-cache/`, model checkpoints, environment directories, secrets, build artifacts and non-reviewed local screenshots are ignored. Download receipts and reviewed source metadata are included as historical provenance; a fresh clone does not contain the developer's private archive. No Git history from unrelated learning-note projects is included.

Use [SECURITY.md](../SECURITY.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) before accepting contributions or exposing a service. Both Studio and the experimental Engine are intended for explicit local/SSH-forwarded, single-owner workflows.
