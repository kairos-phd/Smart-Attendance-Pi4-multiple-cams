import unittest

import numpy as np

from pi4_attendance.quantization import cosine_int8, dequantize_embedding, quantize_embedding


class QuantizationTests(unittest.TestCase):
    def test_int8_embedding_preserves_direction(self) -> None:
        rng = np.random.default_rng(42)
        embedding = rng.normal(size=512).astype(np.float32)
        quantized, scale = quantize_embedding(embedding)
        recovered = dequantize_embedding(quantized, scale)
        reference = embedding / np.linalg.norm(embedding)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertGreater(float(np.dot(reference, recovered)), 0.999)
        self.assertGreater(cosine_int8(quantized, scale, quantized, scale), 0.9999)

    def test_int8_similarity_separates_unrelated_vectors(self) -> None:
        left, left_scale = quantize_embedding(np.array([1.0, 0.0, 0.0]))
        right, right_scale = quantize_embedding(np.array([0.0, 1.0, 0.0]))
        self.assertLess(abs(cosine_int8(left, left_scale, right, right_scale)), 0.001)


if __name__ == "__main__":
    unittest.main()
