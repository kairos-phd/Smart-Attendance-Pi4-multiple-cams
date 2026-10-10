# Share existing Pi data with the website

`GET /api/dashboard` is an admin-authenticated, read-only projection of the existing SQLite store. It includes safe user metadata, saved grouped attendance, rejected unknown observations, and runtime camera health. It excludes face embeddings, candidate identities for unknown observations, and images. No inference model, registration or database migration is performed.

Dates are interpreted as inclusive Asia/Ho_Chi_Minh calendar days. `after_attendance_id` and `after_unknown_rowid` are independent cursors; each response returns a next cursor or null when exhausted. Send -1 for a feed already exhausted while paging the other. The maximum page size is 500. A recognition attempt suppressed by attendance cooldown is not another saved attendance event.

For an existing Pi with locally working INT8 inference, use `scripts/update_dashboard_only.py --repo <repo> --source <dashboard_feed.py>` to inspect and then add `--apply`. It installs the feed module and patches only the final return of the existing API factory, with backups and rollback on interruption. It does not replace inference or model code. Restart the attendance service afterward; capture and recognition pause during restart.

The webserver repository includes the complete Pi/Windows commands in [docs/PI_DASHBOARD_DATA.md](https://github.com/kairos-phd/Smart-Attendance-Pi4-Webserver/blob/main/docs/PI_DASHBOARD_DATA.md). Enable `VITE_DATA_SOURCE=pi` in the website build and use the authenticated lightweight gateway. Existing Firebase login accounts remain separate from Pi face identities. This shares live Pi data rather than uploading SQLite history into Firestore.
