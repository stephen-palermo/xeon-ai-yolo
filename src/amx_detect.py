#!/usr/bin/env python3
"""Detect Intel AMX capabilities (AMX-TILE, AMX-BF16, AMX-INT8) on Intel CPUs.

This is a pure-Python, container-friendly re-implementation of the detection
logic from https://github.com/intel/Intel-AMX-Detection.

The reference project queries the processor with the ``CPUID`` instruction and
inspects leaf 7, sub-leaf 0, register EDX:

    CPUID.07H.00H:EDX[22]  ->  AMX-BF16  (BFloat16 operations)
    CPUID.07H.00H:EDX[24]  ->  AMX-TILE  (tile architecture foundation)
    CPUID.07H.00H:EDX[25]  ->  AMX-INT8  (8-bit integer operations)

OS enablement is reported by the XCR0 control register (read with XGETBV);
bits 17 (XTILECFG) and 18 (XTILEDATA) must be set for the OS to manage AMX
state via XSAVE.

On Linux (the common case for Kubernetes nodes) the kernel already exposes
these capabilities through ``/proc/cpuinfo`` feature flags
(``amx_tile``, ``amx_bf16``, ``amx_int8``), so we use that as the primary,
dependency-free source and fall back to the CPUID instruction where possible.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Optional

# CPUID.07H.00H:EDX feature bit positions (per Intel SDM / reference project).
_BIT_AMX_BF16 = 22
_BIT_AMX_TILE = 24
_BIT_AMX_INT8 = 25

# XCR0 bit positions for OS-managed AMX state.
_BIT_XTILECFG = 17
_BIT_XTILEDATA = 18


@dataclass
class AmxStatus:
    """Result of an AMX capability probe."""

    amx_tile: bool = False
    amx_bf16: bool = False
    amx_int8: bool = False
    amx_os_enabled: bool = False
    source: str = "unknown"

    @property
    def cpu_supported(self) -> bool:
        """True when the CPU advertises every AMX sub-feature."""
        return self.amx_tile and self.amx_bf16 and self.amx_int8

    @property
    def usable(self) -> bool:
        """True when the CPU supports AMX *and* the OS has enabled the state."""
        return self.cpu_supported and self.amx_os_enabled


def _detect_via_proc_cpuinfo() -> Optional[AmxStatus]:
    """Detect AMX by parsing Linux ``/proc/cpuinfo`` feature flags."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return None

    flags: set[str] = set()
    for line in content.splitlines():
        if line.lower().startswith("flags") and ":" in line:
            flags.update(line.split(":", 1)[1].split())
            break

    if not flags:
        return None

    return AmxStatus(
        amx_tile="amx_tile" in flags,
        amx_bf16="amx_bf16" in flags,
        amx_int8="amx_int8" in flags,
        # The kernel only exposes the amx_* flags after it has enabled the
        # AMX state for user space, so their presence implies OS support.
        amx_os_enabled=any(f in flags for f in ("amx_tile", "amx_bf16", "amx_int8")),
        source="/proc/cpuinfo",
    )


def _read_cpuid_leaf7() -> Optional[int]:
    """Return EDX from CPUID leaf 7, sub-leaf 0 using ctypes, or None.

    Builds and runs a tiny machine-code stub. Works on x86-64 Linux/Windows
    where the process is allowed to allocate executable memory. Returns None on
    any failure so callers can fall back to other detection methods.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return None

    import ctypes  # noqa: PLC0415 - imported lazily, only needed on x86-64

    # mov r10, rcx ; (Win) or mov r10, rdi (SysV) -> we pass leaf/subleaf in regs
    # Portable approach: assemble code that sets eax=7, ecx=0, runs cpuid and
    # returns edx in eax. Calling convention differs between SysV and Windows,
    # so we save the destination pointer and write all four registers.
    is_windows = platform.system() == "Windows"

    if is_windows:
        # Args (Win64): rcx = pointer to uint32[4] output buffer.
        code = bytes(
            (
                0x53,                          # push rbx
                0x49, 0x89, 0xC9,              # mov r9, rcx (save out ptr)
                0xB8, 0x07, 0x00, 0x00, 0x00,  # mov eax, 7
                0x31, 0xC9,                    # xor ecx, ecx
                0x0F, 0xA2,                    # cpuid
                0x41, 0x89, 0x01,              # mov [r9+0], eax
                0x41, 0x89, 0x59, 0x04,        # mov [r9+4], ebx
                0x41, 0x89, 0x49, 0x08,        # mov [r9+8], ecx
                0x41, 0x89, 0x51, 0x0C,        # mov [r9+12], edx
                0x5B,                          # pop rbx
                0xC3,                          # ret
            )
        )
    else:
        # Args (SysV): rdi = pointer to uint32[4] output buffer.
        code = bytes(
            (
                0x53,                          # push rbx
                0x49, 0x89, 0xF9,              # mov r9, rdi (save out ptr)
                0xB8, 0x07, 0x00, 0x00, 0x00,  # mov eax, 7
                0x31, 0xC9,                    # xor ecx, ecx
                0x0F, 0xA2,                    # cpuid
                0x41, 0x89, 0x01,              # mov [r9+0], eax
                0x41, 0x89, 0x59, 0x04,        # mov [r9+4], ebx
                0x41, 0x89, 0x49, 0x08,        # mov [r9+8], ecx
                0x41, 0x89, 0x51, 0x0C,        # mov [r9+12], edx
                0x5B,                          # pop rbx
                0xC3,                          # ret
            )
        )

    try:
        if is_windows:
            return _run_machine_code_windows(code)
        return _run_machine_code_posix(code)
    except Exception:  # pragma: no cover - hardware/OS dependent
        return None


def _run_machine_code_posix(code: bytes) -> Optional[int]:
    import ctypes  # noqa: PLC0415
    import mmap  # noqa: PLC0415

    buf = mmap.mmap(
        -1,
        len(code),
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
    )
    buf.write(code)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))

    out = (ctypes.c_uint32 * 4)()
    func_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))
    func = func_type(addr)
    func(out)
    buf.close()
    return int(out[3])  # EDX


def _run_machine_code_windows(code: bytes) -> Optional[int]:
    import ctypes  # noqa: PLC0415

    kernel32 = ctypes.windll.kernel32
    MEM_COMMIT_RESERVE = 0x3000
    PAGE_EXECUTE_READWRITE = 0x40

    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    ptr = kernel32.VirtualAlloc(
        None, len(code), MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE
    )
    if not ptr:
        return None
    ctypes.memmove(ptr, code, len(code))

    out = (ctypes.c_uint32 * 4)()
    func_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))
    func = func_type(ptr)
    func(out)
    kernel32.VirtualFree(ctypes.c_void_p(ptr), 0, 0x8000)  # MEM_RELEASE
    return int(out[3])  # EDX


def _detect_via_cpuid() -> Optional[AmxStatus]:
    """Detect AMX directly via the CPUID instruction."""
    edx = _read_cpuid_leaf7()
    if edx is None:
        return None

    return AmxStatus(
        amx_bf16=bool(edx & (1 << _BIT_AMX_BF16)),
        amx_tile=bool(edx & (1 << _BIT_AMX_TILE)),
        amx_int8=bool(edx & (1 << _BIT_AMX_INT8)),
        # CPUID alone cannot prove OS enablement; assume false here and let the
        # /proc/cpuinfo path (preferred on Linux) report OS state when present.
        amx_os_enabled=False,
        source="cpuid",
    )


def detect_amx() -> AmxStatus:
    """Detect AMX capabilities using the best available method.

    Order of preference:
        1. ``/proc/cpuinfo`` flags (Linux containers / K8S nodes).
        2. The ``CPUID`` instruction via a small machine-code stub.
        3. An empty (all-False) status when nothing is available.
    """
    status = _detect_via_proc_cpuinfo()
    if status is not None and (status.cpu_supported or any(
        (status.amx_tile, status.amx_bf16, status.amx_int8)
    )):
        return status

    cpuid_status = _detect_via_cpuid()
    if cpuid_status is not None:
        # Preserve OS-enabled info from /proc/cpuinfo if we had it.
        if status is not None:
            cpuid_status.amx_os_enabled = status.amx_os_enabled
        return cpuid_status

    return status or AmxStatus(source="unavailable")


def format_report(status: AmxStatus) -> str:
    """Render a human-readable AMX report."""
    def yn(value: bool) -> str:
        return "Yes" if value else "No"

    lines = [
        "Checking Intel AMX support...",
        "",
        f"Detection source: {status.source}",
        "",
        "AMX Support Status:",
        f"  AMX-TILE (Tile Architecture):  {yn(status.amx_tile)}",
        f"  AMX-BF16 (BFloat16):           {yn(status.amx_bf16)}",
        f"  AMX-INT8 (8-bit Integer):      {yn(status.amx_int8)}",
        f"  AMX State enabled by OS:       {yn(status.amx_os_enabled)}",
        "",
        f"AMX Support in CPU: {yn(status.cpu_supported)}",
        f"AMX usable now:     {yn(status.usable)}",
    ]
    if status.cpu_supported:
        lines += [
            "",
            "Your CPU supports all AMX features!",
            "This includes:",
            "  - Tile Architecture (AMX-TILE)",
            "  - BFloat16 Operations (AMX-BF16)",
            "  - 8-bit Integer Operations (AMX-INT8)",
        ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect Intel AMX (AMX-TILE/BF16/INT8) CPU capabilities."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero if full AMX CPU support is not present.",
    )
    args = parser.parse_args(argv)

    status = detect_amx()

    if args.json:
        payload = asdict(status)
        payload["cpu_supported"] = status.cpu_supported
        payload["usable"] = status.usable
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(status))

    if args.require and not status.cpu_supported:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
