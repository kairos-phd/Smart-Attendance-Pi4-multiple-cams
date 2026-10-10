from types import SimpleNamespace
import sys

from pi4_attendance.tools import embeddings


def test_embedding_export_uses_runtime_identity(monkeypatch, tmp_path):
    called = []
    class Engine:
        model_version = 'actual-runtime-contract'
        def __init__(self, *a, **kw): called.append(kw)
        def start(self): pass
    monkeypatch.setattr(embeddings, 'FaceEngine', Engine, raising=False)
    monkeypatch.setattr(embeddings, 'PiStore', lambda _: SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(embeddings, 'export_npz', lambda store, path, key: called.append(key) or 0)
    monkeypatch.setattr(sys, 'argv', ['embeddings', 'export', str(tmp_path/'backup.npz'),
        '--data-root', str(tmp_path), '--model-root', str(tmp_path), '--model-pack', 'pack',
        '--recognizer-model', 'external.onnx'])
    embeddings.main()
    assert called[-1] == 'actual-runtime-contract'
    assert called[0]['recognizer_model'] == 'external.onnx'
