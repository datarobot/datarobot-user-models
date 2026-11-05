"""Stop the process when the FIPS crypto module stops working.

FIPS 140-3 requires that a module which fails a self-test enters an error state and the application halts: it must
stop serving traffic and log a critical error. In Python the OpenSSL error state only makes hashlib and ssl calls
raise, and frameworks turn those into 500s or failed connections while the process keeps serving. This module detects
the error state and exits with the same FATAL line and exit code as the Java and Go services (DS-3101).

It only acts when OpenSSL is in FIPS mode (fips_mode()). The same image run without FIPS mode gets no checks and no
halts, so commercial installs are unaffected.
"""

# ruff: noqa: PLW0603, PLC0415 - module-level state and lazy imports are how this one-file guard works

import ctypes
import faulthandler
import hashlib
import hmac
import logging
import os
import re
import ssl
import sys
import threading
from typing import Any, Callable, Optional

EXIT_CODE = 78  # EX_CONFIG, as in the Java and Go services

_exit_code: int = EXIT_CODE
_halting = threading.Lock()
_checking = threading.Lock()  # held by check_or_halt, and across fork() (see _install_hooks)
_active = False
_started_pid: Optional[int] = None
_hooks_installed = False
_before_halt: Optional[Callable[[], None]] = None
_lib: Any = None
_prov: Any = None
_interval = 5.0


def active() -> bool:
    """True once start() has found FIPS mode on; the integrations check it to decide whether to install anything."""
    return _active


def fips_mode() -> bool:
    """True when OpenSSL's default properties include fips=yes, so hashlib and ssl only use the FIPS provider.

    This is what the FIPS images' openssl.cnf sets. It reads the setting rather than an environment variable, so it
    matches what the process actually does.
    """
    try:
        import _hashlib

        return bool(_hashlib.get_fips_mode())
    except (ImportError, AttributeError, ValueError):
        return False


def grpc_tls_library() -> Optional[str]:
    """Which TLS library grpcio uses: "openssl" for the +fips builds, which link the system libssl and so the FIPS
    provider, "boringssl" for stock wheels, which bundle their own non-FIPS BoringSSL, or None without grpcio.
    """
    try:
        from grpc._cython import cygrpc
    except ImportError:
        return None
    with open(cygrpc.__file__, "rb") as f:
        return "openssl" if re.search(rb"libssl\.so\.\d", f.read()) else "boringssl"


def halt(reason: str) -> None:
    """Log a FATAL line, run start()'s before_halt, and exit, from any thread. Never returns; no cleanup or atexit
    handlers run.

    The line goes to the process's real stderr (sys.__stderr__), not sys.stderr, which Celery may have redirected to
    a logger.
    """
    if _halting.acquire(blocking=False):
        try:
            stream = sys.__stderr__ or sys.stderr
            stream.write(f"[datarobot-fips] FATAL: FIPS crypto module error, halting: {reason}\n")
            stream.flush()
            if _before_halt is not None:
                _before_halt()
        finally:
            os._exit(_exit_code)
    threading.Event().wait()  # another thread is already exiting


class _OsslParam(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_char_p),
        ("data_type", ctypes.c_uint),
        ("data", ctypes.c_void_p),
        ("data_size", ctypes.c_size_t),
        ("return_size", ctypes.c_size_t),
    ]


_OSSL_PARAM_INTEGER = 1


def _fips_provider() -> tuple[Any, Any]:
    """Return (libcrypto, provider) for the FIPS provider that ssl and hashlib use, or (None, None) without one."""
    # Only the libcrypto that ssl and hashlib already use, never a second copy: RTLD_NOLOAD fails unless the library
    # is already loaded (importing ssl above loads it), and the major version comes from ssl. A plain
    # CDLL("libcrypto.so.3") would load OpenSSL 3 into an OpenSSL-4 process when the image also ships libcrypto.so.3.
    try:
        lib = ctypes.CDLL(f"libcrypto.so.{ssl.OPENSSL_VERSION_INFO[0]}", mode=os.RTLD_NOLOAD)
    except OSError:
        return None, None  # _ssl is linked statically or against a renamed libcrypto
    lib.OpenSSL_version_num.restype = ctypes.c_ulong
    if lib.OpenSSL_version_num() != ssl.OPENSSL_VERSION_NUMBER:  # both read at run time
        return None, None
    lib.OSSL_PROVIDER_available.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.OSSL_PROVIDER_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.OSSL_PROVIDER_load.restype = ctypes.c_void_p
    lib.OSSL_PROVIDER_get_params.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    if not lib.OSSL_PROVIDER_available(None, b"fips"):
        return None, None
    # The provider is already loaded by openssl.cnf; this takes another reference to it.
    return lib, lib.OSSL_PROVIDER_load(None, b"fips")


def _provider_status() -> Optional[int]:
    """The FIPS provider's own "status" parameter: 1 while running, 0 in its error state, None if it can't be read.

    The provider reports its error state as a successful query that returns 0. A failed query, or one that leaves the
    value unset, only means the status can't be read, so check() relies on the known-answer tests instead of halting.
    """
    value = ctypes.c_int(-1)
    params = (_OsslParam * 2)()
    params[0] = _OsslParam(
        b"status",
        _OSSL_PARAM_INTEGER,
        ctypes.cast(ctypes.byref(value), ctypes.c_void_p),
        ctypes.sizeof(value),
        0,
    )
    if not _lib.OSSL_PROVIDER_get_params(_prov, params) or value.value == -1:
        return None
    return value.value


def check() -> Optional[str]:
    """Return None if the module works, otherwise why not. Takes a few microseconds.

    Reads the FIPS provider's status, then runs known-answer tests for SHA-256 and HMAC-SHA-256 and reads from the
    DRBG. A module in its error state fails every operation, so either part detects it; the tests still run if the
    provider can't be looked up or its status can't be read.
    """
    try:
        if _prov and _provider_status() not in (1, None):
            return "OpenSSL FIPS provider status is not 1 (error state)"
        sha256_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        if hashlib.sha256(b"abc").hexdigest() != sha256_abc:
            return "SHA-256 known-answer test failed"
        # RFC 4231 test case 1.
        mac = hmac.new(b"\x0b" * 20, b"Hi There", "sha256").hexdigest()
        if mac != "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7":
            return "HMAC-SHA-256 known-answer test failed"
        ssl.RAND_bytes(32)
    except Exception as e:  # noqa: BLE001 - any failure here means the module is unusable
        return f"{type(e).__name__}: {e}"
    return None


def check_or_halt() -> None:
    """Halt if check() fails. Does nothing outside FIPS mode, so every hook is a no-op there."""
    if not _active:
        return
    with _checking:
        reason = check()
    if reason:
        halt(reason)  # outside the lock, so a slow before_halt doesn't hold off fork()


def on_error(exc: Optional[BaseException] = None) -> None:
    """Call wherever crypto raised (ssl.SSLError, a hashlib ValueError, a failed TLS call). In FIPS mode it halts if the
    module is in its error state; otherwise it returns, so ordinary failures like a bad client certificate are
    unaffected."""
    check_or_halt()


class _ErrorLogHandler(logging.Handler):
    """Checks the module whenever anything logs an exception at ERROR or above. Frameworks log the exceptions they
    swallow (a 500, a failed handshake, a failed task), so this catches most of them without wiring each one.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            check_or_halt()


def _install_hooks(loggers: tuple[str, ...]) -> None:
    """Hook uncaught and logged exceptions. Runs once per process tree: a forked child inherits the hooks, so a
    start() in the child must not add them again."""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    # A fork() while the watchdog is inside check() would hand the child OpenSSL locks that no thread will ever
    # release. Hold off forks for the few microseconds a check takes.
    os.register_at_fork(
        before=_checking.acquire,
        after_in_parent=_checking.release,
        after_in_child=_checking.release,
    )

    for name in loggers:
        logging.getLogger(name).addHandler(_ErrorLogHandler())

    prev_excepthook, prev_thread_excepthook = sys.excepthook, threading.excepthook

    def excepthook(*args: Any) -> None:
        check_or_halt()
        prev_excepthook(*args)

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        check_or_halt()
        prev_thread_excepthook(args)

    sys.excepthook, threading.excepthook = excepthook, thread_excepthook


def start(
    interval: float = 5.0,
    exit_code: int = EXIT_CODE,
    loggers: tuple[str, ...] = ("", "uvicorn.error", "gunicorn.error"),
    bundled_grpc: str = "halt",
    before_halt: Optional[Callable[[], None]] = None,
) -> bool:
    """Do nothing and return False unless OpenSSL is in FIPS mode. In FIPS mode, check once, halting on failure, then
    again every interval seconds on a daemon thread, hook uncaught exceptions and exceptions logged on `loggers`, and
    return True. ("" is the root logger; the uvicorn and gunicorn loggers do not propagate to it.) Call it in every
    process that serves traffic, after logging is configured, because logging.config.dictConfig drops handlers it did
    not create.

    bundled_grpc is what to do in FIPS mode when grpcio bundles BoringSSL: "halt" (refuse to start, exit_code) or
    "warn" (log and continue, for a service that only uses plaintext gRPC behind a mesh).

    before_halt, if given, is called by halt() after the FATAL line and before the process exits, from whichever
    thread detected the failure. Use it when exiting alone doesn't stop the service, as in a Celery pool child, which
    must stop its parent first. If it raises, the process still exits.

    Calling start() again in the same process does nothing and returns True. Calling it in a process forked after
    start() (a Celery pool child) checks again, starts that process's own watchdog and sets its own exit_code and
    before_halt; the forked process already has the hooks.
    """
    global _exit_code, _active, _lib, _prov, _started_pid, _before_halt, _interval
    if not fips_mode():
        sys.stderr.write("[datarobot-fips] FIPS mode is off; crypto halt checks disabled\n")
        return False
    if _started_pid == os.getpid():
        return True
    _started_pid = os.getpid()
    _exit_code = exit_code
    _before_halt = before_halt
    _active = True
    try:
        _lib, _prov = _fips_provider()
    except Exception:  # noqa: BLE001 - e.g. no such symbols: known-answer tests only
        _lib, _prov = None, None
    if _prov is None:
        sys.stderr.write(
            "[datarobot-fips] WARNING: FIPS provider status unavailable; using known-answer tests only\n",
        )
    faulthandler.enable()  # a crash in a C extension (grpcio-fips) still leaves a traceback in the log
    if grpc_tls_library() == "boringssl":
        reason = "grpcio bundles BoringSSL, not the FIPS module; install the grpcio +fips build"
        if bundled_grpc == "halt":
            halt(reason)
        sys.stderr.write(f"[datarobot-fips] WARNING: {reason}\n")
    check_or_halt()
    sys.stderr.write(
        f"[datarobot-fips] crypto known-answer tests passed; rechecking every {interval:g}s\n",
    )

    _interval = interval
    _start_watchdog()
    _install_hooks(loggers)
    return True


def _start_watchdog() -> None:
    def watchdog() -> None:
        while not threading.Event().wait(_interval):
            check_or_halt()

    threading.Thread(target=watchdog, name="fips-guard", daemon=True).start()


def restart_watchdog(exit_code: Optional[int] = None) -> None:
    """Start the watchdog again in a forked child, which inherits no threads. Runs no check itself: a check in the
    fork window of a multithreaded parent can deadlock on OpenSSL locks, so the first one comes after _interval.
    exit_code replaces start()'s for this process, e.g. 3 in a gunicorn worker. Does nothing outside FIPS mode.
    """
    global _exit_code
    if not _active:
        return
    if exit_code is not None:
        _exit_code = exit_code
    _start_watchdog()
