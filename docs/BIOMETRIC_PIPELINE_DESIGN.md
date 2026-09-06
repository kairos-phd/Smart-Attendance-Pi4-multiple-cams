# Biometric Metadata Pipeline Design

## Goal

Replace cross-camera quorum voting with a realtime, per-observation biometric decision pipeline suitable for Raspberry Pi 4, while preserving attendance cooldown, local biometric ownership, and optional metadata-only cloud synchronization.

## Runtime flow

`Camera capture -> frame validation -> optional motion/rate trigger -> bounded fair queue -> face detection/alignment -> embedding match -> recognition decision -> SQLite attendance -> metadata-only cloud summary`

Camera identity is provenance, not a quorum requirement. One accepted observation can create attendance; the existing global per-user cooldown suppresses repeated events from the same or different cameras.

## Recognition decision

An accepted decision must carry the originating observation ID, camera ID, capture time, user identity, similarity, best-to-second margin, detection score, face quality, bounding box, and model version. A rejected decision carries the same non-identity metadata and a stable rejection reason.

Acceptance requires all configured thresholds: similarity, margin, minimum detection confidence, and minimum face quality. No image or embedding is sent to Firebase.

## Image contract and unknown crops

Live and enrollment images entering inference must be non-empty contiguous `uint8` BGR arrays shaped `H x W x 3`. Detection receives the complete frame. Any retained unknown image is cropped only after detection from a clamped bounding box with bounded padding; an invalid or empty crop is not stored.

## Embeddings

SQLite remains authoritative. Active embeddings are cached in memory and refreshed after enrollment or status changes. Matching is vectorized and restricted to embeddings whose model version matches the active model fingerprint/version.

NumPy files are an administrative interchange format, never a per-frame datastore. `.npz` import uses `allow_pickle=False` and requires schema version, user IDs, student IDs, names, INT8 embeddings, positive finite scales, dimensions, statuses, and model version. Invalid or incompatible archives are rejected before database writes.

## Model policy

`buffalo_sc` remains the default baseline. Candidate packs are selected only through labeled target-Pi evaluation. Detection failures justify testing SCRFD-2.5G; correct detections with identity errors justify testing a larger recognizer such as the ResNet50 pack. No candidate becomes default solely because its file is larger.

## Verification

Automated tests cover decision thresholds, cooldown, cache refresh, image validation/cropping, and safe `.npz` validation. The benchmark reports detector/recognizer end-to-end latency, detection counts, and labeled recognition outcomes when labels are supplied. Hardware performance remains unverified until run on the target Raspberry Pi.
