#!/usr/bin/env python3
"""CI: build an optimized abi3 `nesle` wheel (release + target-cpu + optional PGO).

Cross-platform (Linux/macOS/Windows) -- all platform differences are handled in
Python so the GitHub Actions matrix calls one command per OS.

Layers always applied:
  * [profile.release]  thin LTO + codegen-units=1 + panic=abort  (Cargo.toml)
  * -Ctarget-cpu=<--cpu>                                          (per-target baseline)
  * source-level SIMD preprocess / mapper enum / cache-pad / render-skip

PGO is added ONLY when a workload ROM is available -- via `--rom PATH` or the
base64 `PGO_ROM_B64` env (a GitHub secret). Flow:
  instrument wheel (maturin build) -> pip install -> scripts/pgo_workload.py
  -> llvm-profdata merge -> optimized wheel (maturin build).
Without a ROM it does a single optimized build (no PGO) so CI never fails just
because the (gitignored, copyrighted) ROM is absent.

Note: -Ctarget-cpu=x86-64-v3 requires AVX2 (Haswell/Zen1+, 2013+). Drop to
x86-64-v2 in the workflow matrix for maximum CPU compatibility (at the cost of
the `wide` SIMD preprocess falling back to SSE).
"""
import argparse
import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd, env=None) -> None:
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env, cwd=REPO)


def rustc_host() -> str:
    for line in subprocess.check_output(["rustc", "-vV"], text=True).splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("cannot determine rustc host triple")


def find_llvm_profdata() -> str:
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    exe = "llvm-profdata.exe" if os.name == "nt" else "llvm-profdata"
    path = Path(sysroot) / "lib" / "rustlib" / rustc_host() / "bin" / exe
    if not path.exists():
        raise SystemExit(f"{exe} not found at {path}; add the llvm-tools-preview component")
    return str(path)


def resolve_rom(arg_rom):
    if arg_rom:
        return arg_rom if Path(arg_rom).exists() else None
    b64 = os.environ.get("PGO_ROM_B64")
    if b64:
        dst = REPO / "pgo-rom.nes"
        dst.write_bytes(base64.b64decode(b64))
        return str(dst)
    # In-wheel SMB1 (staged by CI from the private ROM source, or present in a
    # local build) -> representative PGO with no CI secret; override with --rom.
    staged = REPO / "crates" / "nesle-py" / "python" / "nesle" / "roms" / "super-mario-bros.nes"
    return str(staged) if staged.exists() else None


def maturin_build(out: str, target: str, rustflags: str) -> None:
    env = dict(os.environ, RUSTFLAGS=rustflags)
    # --interpreter REQUIRED on Windows with --target (maturin needs a concrete interpreter to version-probe).
    run([sys.executable, "-m", "maturin", "build", "--release",
         "--target", target, "--interpreter", sys.executable, "--out", out], env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--rom", default=None)
    args = ap.parse_args()

    base = f"-Ctarget-cpu={args.cpu}"
    rom = resolve_rom(args.rom)

    if not rom:
        print("== no workload ROM -> optimized build WITHOUT PGO ==", flush=True)
        maturin_build(args.out, args.target, base)
        return

    print(f"== PGO build (workload ROM: {rom}) ==", flush=True)
    pgo = REPO / "pgo-data"
    if pgo.exists():
        shutil.rmtree(pgo)
    pgo.mkdir(parents=True)

    # 1. instrumented wheel -> install into the CI interpreter
    instr = REPO / "build-instr"
    if instr.exists():
        shutil.rmtree(instr)
    maturin_build(str(instr), args.target, f"{base} -Cprofile-generate={pgo}")
    wheels = sorted(instr.glob("*.whl"))
    if not wheels:
        raise SystemExit("instrument build produced no wheel")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheels[0]])

    # 2. run the workload (unique profraw per process)
    env = dict(os.environ,
               LLVM_PROFILE_FILE=str(pgo / "nesle-%p-%m.profraw"),
               NESLE_PGO_ROM=rom)
    run([sys.executable, REPO / "scripts" / "pgo_workload.py", rom], env=env)

    # 3. merge profiles
    profraws = sorted(str(p) for p in pgo.glob("*.profraw"))
    if not profraws:
        raise SystemExit("no .profraw produced by the workload")
    merged = pgo / "merged.profdata"
    run([find_llvm_profdata(), "merge", "-o", str(merged), *profraws])

    # 4. optimized wheel using the merged profile
    maturin_build(args.out, args.target,
                  f"{base} -Cprofile-use={merged} -Cllvm-args=-pgo-warn-missing-function")


if __name__ == "__main__":
    main()
