# microSD data and dataset design

Default authoritative root: `/var/lib/sic-attendance`.

```text
smart_attendance.sqlite3
smart_attendance.sqlite3-wal
smart_attendance.sqlite3-shm
enrollments/<user-id>/original/*.jpg
unknown/YYYY/MM/DD/*.jpg
calibration/*
tmp/*
```

## Representations

- Camera and stored JPEG pixels: `uint8`.
- Face embeddings: signed INT8 BLOB plus per-vector scale and SHA-256 model compatibility key.
- Similarity, margin, confidence, and quality metadata: integer scaled by 10,000 in SQLite.
- Recognition events: observation ID, camera, identity decision, scores, bbox, reason, model key, and timestamps.

Embeddings are never treated as interchangeable across model versions. Users enrolled with an older model version must be re-enrolled or migrated with a validated conversion process.

SQLite remains authoritative. The explicit `sic-pi4-embeddings` command exports or imports a versioned `.npz` archive; runtime inference never scans NumPy files. Imports use `allow_pickle=False`, validate every row before one database transaction, and reject incompatible model fingerprints.

```bash
.venv/bin/sic-pi4-embeddings export backup.npz --data-root /var/lib/sic-attendance --model-root /opt/sic-attendance --model-pack pi4_fp32
.venv/bin/sic-pi4-embeddings import backup.npz --data-root /var/lib/sic-attendance --model-root /opt/sic-attendance --model-pack pi4_fp32
```

## Durability

SQLite uses WAL, foreign keys, a busy timeout, and normal synchronous mode. Images are written to a temporary file and atomically renamed. Use a high-endurance A2 microSD of at least 64 GB and keep the Pi on a stable power supply.

Create regular encrypted backups to a separately controlled device. A microSD failure otherwise destroys the authoritative biometric dataset. Do not put runtime data in the Git checkout.

Unknown face crops are clamped to the detected bbox with bounded padding; full camera frames are not retained by this path. Crops are automatically deleted after `UNKNOWN_RETENTION_DAYS`. Enrollment deletion policy should be agreed with the institution before adding permanent deletion.
