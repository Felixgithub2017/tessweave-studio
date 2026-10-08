# Security

This release is a local, single-owner control plane, not a multi-tenant hosted service. Do not expose it publicly. Treat the session URL as a secret. Only authorize necessary filesystem roots. Backend Python environments, SSH configuration and installed profilers must be trusted.

Model discovery never imports model code or loads pickle weights. Execution invokes real third-party frameworks and therefore expands the trust boundary. Review model provenance, backend packages and licences before starting. No credential vault is implemented. SSH uses the system's existing key/agent and strict host verification.

Logs can contain dataset contents despite basic token redaction. The SQLite state directory and generated artifacts must be treated as confidential. Existing-model inspection hashes metadata only; new managed downloads verify full file hashes. Sandbox containers, signed plugins, RBAC and resource quotas are not implemented.

The public downloader never sends HF tokens, cookies, or passwords to mirrors. It accepts only built-in HTTPS origins and approved CDN suffixes, rejects private-address DNS results, validates repository paths, pins commits, and verifies LFS SHA-256 / Git blob SHA-1. This is not publisher signature verification: trust still depends on the selected metadata source and HTTPS. A compromised allowed source/CDN, DNS rebinding, or a concurrent hostile local filesystem writer is outside this single-owner prototype's hardened threat model. Do not expose it to untrusted users. Run one control-plane process per state directory. Downloads do not execute repository code, extract archives, or overwrite existing model directories. Gated/private models require a future credential-and-license adapter; no access control bypass is provided.

Report security issues privately to the repository owner after publication; no hosted reporting address exists yet. Avoid including credentials or private model/data content in bug reports.
