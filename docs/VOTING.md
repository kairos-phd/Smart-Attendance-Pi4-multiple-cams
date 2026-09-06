# Biometric recognition policy

## Inputs

Every face observation records camera ID, capture time, candidate identity, top-match similarity, best-to-second margin, detection confidence, face quality, bounding box, model compatibility key, and observation ID.

## Acceptance

One observation is accepted only when similarity, margin, detection confidence, and face quality all meet their configured thresholds. Camera ID records provenance; it does not contribute a vote. An accepted observation creates attendance only when the same user is outside `ATTENDANCE_COOLDOWN_SECONDS`.

Rejected observations are persisted as metadata. When retention is enabled, only the clamped detected-face crop is stored. Images and embeddings are never uploaded to Firebase.

## Camera and runtime failures

Each configured camera captures independently into a fair bounded queue. `/health` reports starting, ok, degraded, or stopped state from inference-thread and persisted camera status, together with queue age, inference latency, queue depth, and cumulative dropped frames.

## Anti-spoofing boundary

Threshold recognition does not prove liveness. Add multi-frame head-motion/blink consistency, a validated anti-spoof model, or depth/IR hardware if spoof resistance is an assessed requirement.
