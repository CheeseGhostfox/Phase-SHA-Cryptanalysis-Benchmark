"""Cross-platform C++17 OpenMP builder for Ouroboros-64 physical core."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

CORE_DIR = Path(__file__).resolve().parent
CPP_DIR = CORE_DIR / "cpp"


def find_cxx_compiler() -> str:
    """Locate available C++17 compiler (g++ / clang++)."""
    for name in ("g++", "clang++"):
        exe = shutil.which(name)
        if exe:
            return exe
    raise RuntimeError("No suitable C++17 compiler (g++ or clang++) found in PATH.")


def get_capi_lib_path() -> Path:
    suffix = ".dll" if platform.system().lower() == "windows" else ".so"
    return CORE_DIR / f"_ouroboros_capi{suffix}"


def add_mingw_runtime_dirs() -> None:
    """Ensure Windows MinGW runtime DLLs (libgomp-1.dll, libwinpthread-1.dll) resolve cleanly."""
    if platform.system().lower() != "windows":
        return
    if hasattr(os, "add_dll_directory"):
        seen = set()
        for name in ("g++", "gcc"):
            exe = shutil.which(name)
            if exe:
                bindir = str(Path(exe).resolve().parent)
                if bindir not in seen:
                    seen.add(bindir)
                    try:
                        os.add_dll_directory(bindir)
                    except OSError:
                        pass


def build_capi(force: bool = False) -> Path:
    """Compile c_api.cpp -> _ouroboros_capi.(dll|so) with -O3 -std=c++17 -fopenmp."""
    out_lib = get_capi_lib_path()
    src_c = CPP_DIR / "c_api.cpp"
    src_h = CPP_DIR / "crystal_core.hpp"
    if (
        not force
        and out_lib.exists()
        and out_lib.stat().st_mtime >= max(src_c.stat().st_mtime, src_h.stat().st_mtime)
    ):
        return out_lib

    cxx = find_cxx_compiler()
    is_win = platform.system().lower() == "windows"
    cmd = [
        cxx,
        "-O3",
        "-std=c++17",
        "-fopenmp",
        "-shared",
        "-Wall",
        "-Wextra",
    ]
    if not is_win:
        cmd.append("-fPIC")
    else:
        cmd.extend(["-static-libgcc", "-static-libstdc++"])
    cmd.extend([str(src_c), "-o", str(out_lib)])

    print(f"[Ouroboros build_cpp] Compiling C-ABI core: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"C++ build failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    if res.stderr.strip():
        print(f"[Ouroboros build_cpp] Compiler warnings:\n{res.stderr.strip()}", flush=True)
    return out_lib


def build_pybind_optional() -> Optional[Path]:
    """Compile optional PyBind11 module when pybind11 headers are available."""
    try:
        import pybind11  # type: ignore
        import sysconfig
    except Exception:
        return None

    cxx = find_cxx_compiler()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if platform.system().lower() == "windows" else ".so"
    )
    out_mod = CORE_DIR / f"_ouroboros_pybind{ext_suffix}"
    src_pb = CPP_DIR / "pybind_module.cpp"
    inc_py = sysconfig.get_path("include")
    inc_pb = pybind11.get_include()

    cmd = [
        cxx,
        "-O3",
        "-std=c++17",
        "-fopenmp",
        "-shared",
        f"-I{inc_py}",
        f"-I{inc_pb}",
    ]
    if platform.system().lower() != "windows":
        cmd.append("-fPIC")
    else:
        cmd.extend(["-static-libgcc", "-static-libstdc++"])
    cmd.extend([str(src_pb), "-o", str(out_mod)])
    if platform.system().lower() == "windows":
        py_lib = Path(sys.base_prefix) / "libs"
        if py_lib.exists():
            py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
            cmd.extend([f"-L{py_lib}", f"-l{py_ver}"])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    return out_mod


if __name__ == "__main__":
    lib = build_capi(force=True)
    pb = build_pybind_optional()
    print(f"Built C-ABI: {lib} ({lib.stat().st_size} bytes), PyBind: {pb}", flush=True)
