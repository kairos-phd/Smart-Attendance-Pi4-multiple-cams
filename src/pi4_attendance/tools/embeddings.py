from __future__ import annotations

import argparse
from pathlib import Path

from pi4_attendance.embedding_interchange import export_npz, import_npz
from pi4_attendance.face_engine import FaceEngine
from pi4_attendance.storage import PiStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly import or export the SQLite embedding catalog")
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-pack", required=True)
    parser.add_argument("--recognizer-model", default="auto", help="Match the runtime RECOGNIZER_MODEL selection")
    parser.add_argument("--det-size", type=int, default=320)
    args = parser.parse_args()
    engine = FaceEngine(args.model_root.resolve(), args.model_pack, args.det_size, .45, .06,
                        recognizer_model=args.recognizer_model)
    engine.start()
    store = PiStore(args.data_root)
    try:
        compatibility_key = engine.model_version
        count = export_npz(store, args.archive, compatibility_key) if args.action == "export" else import_npz(store, args.archive, compatibility_key)
    finally:
        store.close()
    print(f"{args.action}ed {count} users")


if __name__ == "__main__":
    main()
