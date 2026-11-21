# Preserving Registered Users Across Model Changes

You can preserve existing `user_id`, `student_id`, and attendance history by regenerating embeddings from verified original enrollment images. Changing the recognizer model or quantizing from FP32 to INT8 requires running the enrollment images through the new recognizer; never rewrite the `model_version` column directly in SQL to force old embeddings to match. Replacing the face detector also requires re-verifying landmark alignment and face quality. Only a change in the embedding storage representation for the exact same model can be dequantized and requantized without re-running inference, and its approximation error must still be evaluated.

The current repository does not provide an automated in-place user ID migration tool: standard enrollment creates a new user record, and catalog `.npz` import does not update embeddings in place while retaining legacy attendance rows. The media-only and dashboard-only updaters leave the model and user catalog untouched. Keep the currently working recognizer active until the following steps are completed:

1. Audit that original enrollment images exist on disk, match their recorded SHA-256 checksums, and have verified identity labels. Any user missing valid original images must be re-captured with consent; never guess or infer identities.
2. Use the new model manifest/fingerprint and the enrollment validation pipeline (`3–8` images per user, single face, detection score `>= 0.60`, face quality `>= 0.35`, same-person consistency `>= 0.55`) to build a staging catalog. Preserve `user_id` and `student_id`; do not insert duplicate users.
3. Evaluate across multiple registered users, unknown holdout subjects, all configured cameras (`icspring`, `jieli`, `sonix`), and varied lighting/distances. Recalibrate `MATCH_THRESHOLD` and `MATCH_MARGIN` on labeled ground-truth data split by session to measure FAR and FRR. Measuring CPU utilization or file size reduction alone is not sufficient to accept a model change.
4. Stop the service briefly (`sudo systemctl stop sic-pi4-attendance`) and take a consistent SQLite backup (using the SQLite backup API to capture WAL state) alongside the model and `.env` configuration. Within a single database transaction, update only `embedding_int8`, `embedding_dim`, `embedding_scale`, `model_version`, and `updated_at`/audit metadata while keeping `users`, `enrollment_images`, and `attendance` history intact. Verify that the loaded catalog matches the new model compatibility fingerprint before restarting the service.
5. If validation fails, restore the database, model files, and configuration together before restarting. Never roll back only one side of the state.

Read-only inventory on the Pi (prints only counts and schema metadata, never personal names or biometric content):

```bash
python3 - <<'PY'
from pathlib import Path
import sqlite3
root = Path('/var/lib/sic-attendance')
database = root / 'smart_attendance.sqlite3'
with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
    print('users:', db.execute('SELECT COUNT(*) FROM users').fetchone()[0])
    print('image schema:', [r[1] for r in db.execute('PRAGMA table_info(enrollment_images)')])
    print('images:', db.execute('SELECT COUNT(*) FROM enrollment_images').fetchone()[0])
    print('model groups:', db.execute('SELECT model_version,COUNT(*) FROM users GROUP BY model_version').fetchall())
PY
```

Inspect the actual on-device SQLite schema before running any migration utility; do not assume the live Pi database or enrollment directory matches an offline source archive. After verifying live streaming and a 30-minute recognition soak with the active model, collect session-separated ground truth before running controlled A/B comparisons against static INT8 or alternative detector/recognizer candidates, followed by offline/outbox retry verification and presentation attack detection (PAD) evaluation.
