from __future__ import annotations

import argparse
from pathlib import Path

from pi4_attendance.embedding_interchange import export_npz, import_npz
from pi4_attendance.model_identity import model_identity
from pi4_attendance.storage import PiStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly import or export the SQLite embedding catalog")
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-pack", required=True)
    args = parser.parse_args()
    store = PiStore(args.data_root)
    try:
        compatibility_key = model_identity(args.model_root, args.model_pack).compatibility_key
        count = export_npz(store, args.archive, compatibility_key) if args.action == "export" else import_npz(store, args.archive, compatibility_key)
    finally:
        store.close()
    print(f"{args.action}ed {count} users")


if __name__ == "__main__":
    main()
