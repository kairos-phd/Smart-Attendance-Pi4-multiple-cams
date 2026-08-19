# Three-camera voting

## Camera geometry

Place one frontal camera and two oblique cameras so a normal passage is visible in at least two views. Avoid three unrelated rooms or non-overlapping views; they cannot provide identity consensus.

## Vote inputs

Every observation records camera ID, candidate identity, top-match similarity, best-to-second-best margin, detection confidence, face quality, bounding box, and timestamp. Values in SQLite are scaled integers where practical.

## Confirmation

An identity must exceed both `MATCH_THRESHOLD` and `MATCH_MARGIN` on at least `VOTE_REQUIRED_CAMERAS` distinct cameras. Multiple frames from one camera are reduced to that camera's strongest quality-weighted observation.

The coordinator waits until `VOTE_WINDOW_SECONDS` closes before deciding. It rejects insufficient agreement and explicitly marks multiple identities reaching quorum as a conflict.

## Camera failures

- Three online: normal 2-of-3 mode.
- Two online: 2-of-2 is technically possible but is a degraded mode and must be displayed as such by the dashboard.
- One online: never automatically confirm attendance.

The runtime exposes camera health and dropped-frame counters. A production dashboard should use them to enforce or display degraded status.

## Anti-spoofing boundary

Voting reduces single-camera mistakes; it does not prove liveness. A photograph shown to multiple cameras can still attack a purely visual recognizer. Add multi-frame head-motion/blink consistency, a validated anti-spoof model, or depth/IR hardware if spoof resistance is an assessed requirement.

