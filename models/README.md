# Model artifacts

Model weights are intentionally not committed. InsightFace's downloadable pretrained packs are restricted to non-commercial research use, and the generated files are device/build artifacts.

On the development machine or Pi, run:

```bash
sic-pi4-prepare-models \
  --model-root /opt/sic-attendance \
  --calibration-dir /var/lib/sic-attendance/calibration \
  --accept-insightface-noncommercial-license
```

`MODEL_ROOT` is the InsightFace root. Packs are stored under `MODEL_ROOT/models/<pack>`, so this creates:

- `models/pi4_fp32`: graph-optimized FP32 baseline;
- `models/pi4_int8`: statically calibrated QDQ INT8 candidate;
- a manifest containing model versions, sizes, and SHA-256 checksums.

Do not select `pi4_int8` until `sic-pi4-benchmark` passes on the target Raspberry Pi and recognition thresholds have been recalibrated.

