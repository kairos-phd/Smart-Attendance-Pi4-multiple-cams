# Biometric Metadata Pipeline Implementation Plan

**Goal:** Implement the approved per-observation biometric attendance pipeline without replacing the user's existing security hardening changes.

**Architecture:** Keep concurrent camera capture and one bounded inference consumer. Replace temporal cross-camera voting with an immediate threshold policy, persist complete recognition metadata, cache compatible embeddings in RAM, and keep SQLite authoritative.

**Tech Stack:** Python 3.11, OpenCV, NumPy, InsightFace, SQLite, FastAPI, pytest.

**Spec:** `docs/BIOMETRIC_PIPELINE_DESIGN.md`

## Global Constraints

- Preserve existing uncommitted changes in `api.py`, `service.py`, and `storage.py`.
- Do not upload images or embeddings to Firebase.
- Do not change the default pretrained model without target-Pi benchmark evidence.
- Write a failing behavioral test before each production behavior change.

### Task 1: Recognition metadata policy

- Add threshold-based per-observation decisions carrying full biometric metadata.
- Remove camera quorum from runtime configuration and attendance behavior.
- Persist recognition events and retain per-user cooldown.
- Replace voting tests with recognition-policy and storage tests.

### Task 2: Image boundary and safe crop

- Validate live/enrollment frames as non-empty contiguous `uint8` BGR arrays.
- Crop retained unknown faces from a clamped detector bbox with bounded padding.
- Add tests for invalid frames, out-of-bounds boxes, and empty crops.

### Task 3: Embedding cache and NumPy interchange

- Cache compatible active embeddings and refresh after enrollment/status changes.
- Vectorize top-two cosine matching while preserving INT8 embedding storage.
- Add strict, atomic `.npz` import/export with `allow_pickle=False`.
- Add cache and malformed/incompatible archive tests.

### Task 4: Metrics, benchmark, and documentation

- Record queue age and inference latency plus existing dropped-frame counters.
- Update API health, architecture, storage, deployment, model, and README language.
- Extend benchmark output for model selection without changing the default model.
- Run the complete test suite and compile check.
