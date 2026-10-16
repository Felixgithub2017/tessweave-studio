# Bundled deployment-reference catalog

Studio can be used without a sibling Engine checkout. Its fallback snapshot was
copied from the local TessWeave Engine working tree on 2026-10-16; that source was
not yet a published commit. We therefore pin content hashes, not a fictitious tag.

Source project: https://github.com/Felixgithub2017/tessweave-engine

| File | SHA-256 |
| --- | --- |
| Source `flow_engine/deployment_catalog.py` | `7fd49b98e5a046bd0a8aed95467de953ac69a9c985ce870476ad1997ea1259d4` |
| Bundled `workbench/bundled_deployment_catalog.py` | `3088db27e507e8e8c8afc80ea03edda69a751ff2708db505e506a0bb1102d667` |
| `workbench/knowledge/deployment-models-20261015.json` | `2c04f89bc080cc4988517b62616dcee33b43e43edbc05992518632c3d6ff4dc0` |
| `workbench/knowledge/deployment-architecture-references.json` | `3b288945f6bdee756d3f87859ca49efc9cfcdd0290f1d7fe4e12816b00cb3c93` |

The only source adaptation changes the data directory from `data` to `knowledge`.
The [original MIT license](engine-catalog-LICENSE.txt) is preserved. This is a
snapshot of TessWeave's reference catalog, not transplanted vLLM/SGLang kernels.

Resolution order: installed Engine catalog → sibling Engine checkout → bundled
snapshot. An external Engine version can therefore differ from this snapshot.
The tutorials describe external backends and candidate workflows, never certify
device compatibility or start processes. Model configuration sources and revisions
remain attached to each record; upstream model/runtime terms still apply.

## Screenshot reproduction

`tests/test_workflows_browser.cjs` supports `HOME_SCREENSHOT`,
`MODEL_CHOOSER_SCREENSHOT` and `PLANNER_SCREENSHOT` output paths. It requires an
existing Playwright installation and browser; set `PYTHON` and `CHROME_PATH` for
your environment. The fixture uses isolated temporary workspaces and a public
demo hostname. It tests UI navigation and estimation without launching workloads.
No session URLs or tokens should be included in public screenshots.
