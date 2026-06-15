#!/usr/bin/env python3
"""CI: stage private ROMs into the package ROM home before ``maturin build``.

The public repo never commits copyrighted ROMs (the ale-py model: ROMs ship in
the wheel, not in the source tree). CI checks out a private ROM source and this
script copies its ``*.nes`` into ``crates/nesle-py/python/nesle/roms/`` so
maturin's ``include`` bundles them into the wheel. Cross-platform and no-op-safe:
an empty/absent source just stages zero ROMs and the wheel is built ROM-free.

Usage: ``python scripts/stage_roms.py <src-dir> <package-roms-dir>``
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: stage_roms.py <src-dir> <package-roms-dir>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        print(f"stage_roms: source {src} absent -> 0 ROMs staged (ROM-free wheel)")
        return
    roms = sorted(src.rglob("*.nes"))
    for rom in roms:
        shutil.copy2(rom, dst / rom.name)
    print(f"stage_roms: staged {len(roms)} ROM(s) from {src} -> {dst}")


if __name__ == "__main__":
    main()
