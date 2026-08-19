# Model preparation and acceptance

## Packs

The preparation tool acquires the InsightFace `buffalo_sc` source pack only after explicit license acknowledgement, then produces:

- `pi4_fp32`: ONNX graph-preprocessed FP32 baseline;
- `pi4_int8`: QDQ static INT8 detector/recognizer candidate using representative calibration images.

Auxiliary landmark/attribute models remain optimized FP32. A manifest records each model file's SHA-256 hash, size, source pack, and version.

## Calibration data

Use consented images that represent the actual cameras, lighting, distance, face sizes, angles, glasses, and expected users. Do not use only clean portrait photographs. At least 20 images are enforced; 100–300 is preferred.

Calibration data remains under `/var/lib/sic-attendance/calibration` and must not enter Git.

## Acceptance gate

The automated benchmark compares detection count, end-to-end latency, and embedding agreement. Its default candidate gate requires:

- mean embedding cosine at least `0.98` versus FP32;
- no more than approximately 1% detector-count loss;
- at least 5% median latency improvement.

This is only a technical gate. Before selecting INT8, also run labeled verification trials and recalibrate `MATCH_THRESHOLD` and `MATCH_MARGIN`. Measure false accepts, false rejects, and 2-of-3 decision accuracy.

If INT8 fails accuracy or is not faster on the Cortex-A72, keep optimized FP32. Quantization is an option, not a required outcome.

## Model licensing

InsightFace code and downloaded model packs have different terms. The open model packs are described as non-commercial research assets. Do not commit downloaded or converted weights to a public repository without confirming redistribution rights. Commit the reproducible preparation tool and manifests instead.

