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

Biometric threshold matching is not liveness detection. Do not represent the system as spoof-proof without separate anti-spoof testing.

Enrollment accepts 3–8 static JPEG, PNG, or WebP images, up to 8 MiB each and 32 MiB combined. Before OpenCV decoding, the handler checks every compressed file size, MIME/signature agreement, image header dimensions (at most 4096 per axis and 4,194,304 pixels per image), and the 16,777,216-pixel batch limit. Animated WebP and inconsistent WebP canvas/frame sizes are rejected. Decoded dimensions must agree with the header, allowing an axis swap for EXIF orientation.

FastAPI/Starlette parses and may spool multipart files before this handler runs. These checks bound application reads and image decoding; they do not reject large bodies before framework spooling. Configure a request-body size limit at the reverse proxy to bound incoming multipart traffic, including multipart overhead.
