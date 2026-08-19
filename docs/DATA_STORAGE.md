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
- Face embeddings: signed INT8 BLOB plus per-vector scale and model version.
- Similarity, margin, confidence, and quality metadata: integer scaled by 10,000 in SQLite.
- Bounding boxes and camera lists: compact JSON metadata.

Embeddings are never treated as interchangeable across model versions. Users enrolled with an older model version must be re-enrolled or migrated with a validated conversion process.

## Durability

SQLite uses WAL, foreign keys, a busy timeout, and normal synchronous mode. Images are written to a temporary file and atomically renamed. Use a high-endurance A2 microSD of at least 64 GB and keep the Pi on a stable power supply.

Create regular encrypted backups to a separately controlled device. A microSD failure otherwise destroys the authoritative biometric dataset. Do not put runtime data in the Git checkout.

Unknown images are automatically deleted after `UNKNOWN_RETENTION_DAYS`. Enrollment deletion policy should be agreed with the institution before adding permanent deletion.

