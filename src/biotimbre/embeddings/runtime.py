"""Whether onnxruntime can be loaded here, asked of a child process first.

On Windows, onnxruntime needs a recent Microsoft Visual C++ runtime, which
Windows does not keep up to date by itself. With an older ``msvcp140.dll``,
importing onnxruntime fails while its DLL initialises: some releases raise an
``ImportError``, others take the whole process down with an access violation,
which leaves no message at all. For the interface, that means a console that
prints its address and closes.

A crash cannot be caught, only avoided, so the import is first tried in a
child process. Its exit code says whether importing it here is safe, and
when it is not, a message can say what to do instead.
"""

from __future__ import annotations

import functools
import subprocess
import sys

#: The current Visual C++ Redistributable for x64, from Microsoft.
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

#: Seconds the child process gets to import onnxruntime. Normally well under
#: one; the margin is for a first import on a slow disk or an antivirus scan.
CHECK_TIMEOUT = 120.0


def _describe(returncode: int) -> str:
    """An exit code as Windows reports it: NTSTATUS codes in hex."""
    if returncode < 0:
        return f"0x{returncode & 0xFFFFFFFF:08X}"
    return str(returncode)


@functools.cache
def onnxruntime_problem() -> str | None:
    """Why onnxruntime cannot be loaded in this process, if it cannot.

    Only checked on Windows, where the failure can be a crash; elsewhere a
    broken install raises an ordinary ``ImportError``, which the backend
    already reports. Asked once per process.

    Returns:
        A message for the user, or ``None`` when onnxruntime loads, or when
        the check itself could not run, so the real import decides.
    """
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import onnxruntime"],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        return None
    return (
        "onnxruntime, which runs the models, cannot be loaded on this computer "
        f"(importing it exits with {_describe(result.returncode)}). This "
        "usually means the Microsoft Visual C++ Redistributable is missing or "
        f"too old: install the current one from {VC_REDIST_URL} and start "
        "biotimbre again."
    )
