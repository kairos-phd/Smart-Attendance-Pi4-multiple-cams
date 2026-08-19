# Security and privacy

- Keep `.env` mode `0600` and service-account JSON outside Git.
- Use Firebase administrator claims in normal operation. Shared-token mode is only for controlled development networks.
- Keep `API_CORS_ORIGINS` as an explicit allow-list; wildcard origins are rejected. Terminate TLS at a trusted reverse proxy if the API is reachable outside a private LAN.
- Restrict `/var/lib/sic-attendance` to the service user and authorized administrators.
- Never expose filesystem paths as unauthenticated static files.
- Do not upload face images, embeddings, calibration data, or datasets to Firebase or GitHub.
- Record administrator enrollment and status changes in the audit log.
- Apply a documented consent, retention, access, correction, and deletion policy for biometric data.
- Protect SSH with keys, disable password login where practical, and keep Raspberry Pi OS updated.
- Treat physical theft of the microSD as a data breach risk. Use an encrypted partition or application-level encryption when required by policy.

Voting is not liveness detection. Do not represent the system as spoof-proof without separate anti-spoof testing.
