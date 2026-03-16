#!/usr/bin/env python3
"""Generate Python protobuf code for the DoclingDocument proto (messages only, no gRPC services)."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "docling_core" / "proto" / "gen"


def ensure_init_files(path: pathlib.Path) -> None:
    for sub in [path] + [p for p in path.rglob("*") if p.is_dir()]:
        init = sub / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")


def main() -> None:
    proto_file = PROTO_DIR / "ai" / "docling" / "core" / "v1" / "docling_document.proto"
    if not proto_file.exists():
        print(f"Proto file not found: {proto_file}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Messages only (no gRPC services), so use --python_out only.
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        str(proto_file),
    ]
    subprocess.check_call(cmd)
    ensure_init_files(OUT_DIR)
    print("Generated docling_document_pb2 in", OUT_DIR)


if __name__ == "__main__":
    main()
