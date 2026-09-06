# Model preparation and acceptance

## Packs

The preparation tool acquires the InsightFace `buffalo_sc` source pack only after explicit license acknowledgement, then produces:

- `pi4_fp32`: ONNX graph-preprocessed FP32 baseline;
- `pi4_int8`: QDQ static INT8 detector plus optimized FP32 recognizer candidate using representative full-camera calibration images.

The recognizer and auxiliary landmark/attribute models remain optimized FP32. Quantizing the recognizer requires a separate calibration set of correctly aligned 112×112 face crops; resizing arbitrary full frames into the recognizer input would calibrate the wrong data distribution. A manifest records each model file's SHA-256 hash, size, role, quantization status, source pack, and version.

## Calibration data

Use consented images that represent the actual cameras, lighting, distance, face sizes, angles, glasses, and expected users. Do not use only clean portrait photographs. At least 20 images are enforced; 100–300 is preferred.

Calibration data remains under `/var/lib/sic-attendance/calibration` and must not enter Git.

## Acceptance gate

The automated benchmark compares detection count, end-to-end latency, and embedding agreement. Its default candidate gate requires:

- mean embedding cosine at least `0.98` versus FP32;
- no more than approximately 1% detector-count loss;
- at least 5% median latency improvement.

This is only a technical gate. Before selecting INT8 or a larger pretrained pack, also run labeled verification trials and recalibrate `MATCH_THRESHOLD` and `MATCH_MARGIN`. Measure false accepts and false rejects using consented images arranged as `dataset/<identity>/*.jpg`. The benchmark only compares embeddings when the reference and candidate detections overlap at IoU >= 0.50, preventing different faces from being compared.

Use `buffalo_sc`/`pi4_fp32` as the Raspberry Pi baseline. If the baseline misses faces, benchmark an SCRFD-2.5G candidate. If faces are detected correctly but identities are confused, benchmark a larger recognizer such as the ResNet50 pack. `buffalo_s` is not an automatic accuracy upgrade over `buffalo_sc`; `buffalo_l` and `antelopev2` require target-Pi latency, memory, thermal, and queue-pressure evidence before selection.

```bash
.venv/bin/sic-pi4-benchmark --model-root /opt/sic-attendance --dataset /var/lib/sic-attendance/benchmark --reference pi4_fp32 --candidate pi4_int8 --match-threshold 0.45
```

If a candidate fails accuracy, is not faster, or causes sustained queue pressure on the Cortex-A72, keep optimized FP32. Quantization and larger pretrained packs are candidates, not required upgrades.

## Model licensing

InsightFace code and downloaded model packs have different terms. The open model packs are described as non-commercial research assets. Do not commit downloaded or converted weights to a public repository without confirming redistribution rights. Commit the reproducible preparation tool and manifests instead.
