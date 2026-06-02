#!/usr/bin/env python3
"""Download recordings from an OLA in file-transfer mode over USB serial.

The OLA enters file-transfer mode when you tap RESET TWICE within the
~2.5 s decision window (the same multi-press mechanism that 3-taps mag-cal
and 4-taps gyro-cal). In transfer mode the device skips IMU/GNSS setup,
brings up the SD card, and listens on USB serial for these commands:

    list                  - enumerate every BOOT_*/*.dat on the card
    get <N>               - stream file at index N (length-prefixed + CRC32)
    info                  - firmware + SD info
    help                  - command list
    exit | reboot         - leave transfer mode

This script wraps the protocol with a friendly interactive prompt:

    $ python serial_transfer.py --port COM5
    [connect, run `list`, render]
    0   10485760 bytes  BOOT_000022/DATA_BOOT_000022_TIME_20260530T092437.dat
    1   10485760 bytes  BOOT_000023/DATA_BOOT_000023_TIME_20260530T094000.dat
    ...
    Which file to download (index, or 'all' or 'q' to quit)? 0
    [progress bar]
    Saved 10485760 bytes -> DATA_BOOT_000022_TIME_20260530T092437.dat
    CRC32 OK (0x12345678)

For non-interactive use, pass --file-index N (or --all) and --output-dir.

Defaults to 1 Mbaud. The OLA's USB connection is bridged by a CH340, which
makes --baud a real UART line rate — must match the firmware constant
BAUD_RATE_USB. 1 Mbaud is the empirically tested-reliable rate on this
hardware; higher rates lose bytes through the CH340 + host driver.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)


READY_TOKEN = b"READY"
LIST_BEGIN = b"LIST_BEGIN"
LIST_END_PREFIX = b"LIST_END"
GET_BEGIN_PREFIX = b"GET_BEGIN"
GET_END_PREFIX = b"GET_END"
ERROR_PREFIX = b"ERROR"


@dataclass
class RemoteFile:
    index: int
    size_bytes: int
    path: str

    @property
    def local_name(self) -> str:
        # We strip the BOOT_NNNNNN/ prefix when saving locally; the filename
        # itself already encodes the boot count.
        return self.path.split("/")[-1]


# ---------------------------------------------------------------------------
# Low-level serial I/O
# ---------------------------------------------------------------------------


def read_line(ser: serial.Serial, timeout_s: float = 30.0) -> bytes:
    """Read one line ending in '\\n'. Returns the line bytes WITHOUT the
    terminator and without any leading '\\r'. Raises TimeoutError on idle.
    """
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"No newline received in {timeout_s:.1f}s. "
                               f"Got so far: {bytes(buf)!r}")
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            # strip optional trailing \r
            if buf.endswith(b"\r"):
                buf = buf[:-1]
            return bytes(buf)
        buf.extend(b)


def send_command(ser: serial.Serial, cmd: str) -> None:
    ser.write(cmd.encode("ascii") + b"\n")
    ser.flush()


def wait_for_ready(ser: serial.Serial, echo: bool = False, timeout_s: float = 60.0) -> list[bytes]:
    """Read lines until we see READY. Returns the lines collected before READY."""
    out: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = read_line(ser, timeout_s=max(1.0, deadline - time.monotonic()))
        if echo:
            print(f"   <<< {line.decode('latin-1', errors='replace')}")
        if line == READY_TOKEN:
            return out
        out.append(line)
    raise TimeoutError("Device did not return READY within timeout")


# ---------------------------------------------------------------------------
# Protocol commands
# ---------------------------------------------------------------------------


def cmd_list(ser: serial.Serial) -> list[RemoteFile]:
    send_command(ser, "list")
    lines = wait_for_ready(ser)
    # Expect: LIST_BEGIN, <N lines: "idx size path">, LIST_END <count>
    files: list[RemoteFile] = []
    in_block = False
    for line in lines:
        if line == LIST_BEGIN:
            in_block = True
            continue
        if line.startswith(LIST_END_PREFIX):
            in_block = False
            continue
        if line.startswith(ERROR_PREFIX):
            raise RuntimeError(f"Device returned: {line.decode('latin-1')}")
        if in_block and line:
            # idx <space> size <space> path-with-possible-spaces
            parts = line.decode("latin-1").split(" ", 2)
            if len(parts) != 3:
                continue
            idx_s, size_s, path = parts
            try:
                files.append(RemoteFile(int(idx_s), int(size_s), path))
            except ValueError:
                continue
    return files


def cmd_get(ser: serial.Serial, index: int, output_path: Path,
            show_progress: bool = True) -> tuple[int, int]:
    """Download file at `index` to `output_path`. Returns (bytes_received, crc32).
    Raises RuntimeError on protocol/CRC errors."""
    send_command(ser, f"get {index}")

    # Header line.
    header = read_line(ser, timeout_s=15.0)
    if header.startswith(ERROR_PREFIX):
        # consume trailing READY
        wait_for_ready(ser)
        raise RuntimeError(f"Device: {header.decode('latin-1')}")
    if not header.startswith(GET_BEGIN_PREFIX):
        wait_for_ready(ser)
        raise RuntimeError(f"Unexpected header: {header!r}")
    m = re.match(rb"GET_BEGIN\s+(\d+)\s+(\d+)$", header)
    if not m:
        wait_for_ready(ser)
        raise RuntimeError(f"Malformed GET_BEGIN: {header!r}")
    advertised_idx = int(m.group(1))
    advertised_bytes = int(m.group(2))
    if advertised_idx != index:
        raise RuntimeError(f"Device returned index {advertised_idx}, expected {index}")

    # Read exactly `advertised_bytes` of raw binary.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crc = 0
    received = 0
    t0 = time.monotonic()
    last_progress = t0
    with output_path.open("wb") as out_f:
        while received < advertised_bytes:
            want = min(64 * 1024, advertised_bytes - received)
            chunk = ser.read(want)
            if not chunk:
                # Briefly nothing — retry; if too long, bail.
                if time.monotonic() - last_progress > 30.0:
                    raise RuntimeError(
                        f"Serial stalled at {received}/{advertised_bytes} bytes"
                    )
                continue
            out_f.write(chunk)
            crc = zlib.crc32(chunk, crc)
            received += len(chunk)
            now = time.monotonic()
            if show_progress and now - last_progress >= 0.5:
                pct = 100.0 * received / advertised_bytes
                rate = received / max(now - t0, 1e-3) / 1024.0
                eta = (advertised_bytes - received) / max(rate * 1024, 1)
                sys.stdout.write(
                    f"\r   {received:>10d} / {advertised_bytes} bytes "
                    f"({pct:5.1f}%, {rate:6.1f} KB/s, ETA {eta:4.0f}s)   "
                )
                sys.stdout.flush()
                last_progress = now
    if show_progress:
        sys.stdout.write("\n")

    # Trailer: a blank line then GET_END <crc-hex>
    # Some devices emit just "GET_END ..." on the next line; consume blanks.
    while True:
        line = read_line(ser, timeout_s=10.0)
        if line == b"":
            continue
        if line.startswith(GET_END_PREFIX):
            crc_hex = line.decode("latin-1").split(" ", 1)[1].strip()
            device_crc = int(crc_hex, 16)
            break
        if line.startswith(ERROR_PREFIX):
            raise RuntimeError(f"Device error: {line.decode('latin-1')}")
        # Anything else: warn but keep going (could be log spam)
        print(f"   (note) unexpected line during trailer: {line!r}")

    # Drain READY
    wait_for_ready(ser)

    if device_crc != crc:
        raise RuntimeError(
            f"CRC32 mismatch! Device says 0x{device_crc:08x}, host computed 0x{crc:08x}. "
            f"File is suspect; do not trust it."
        )
    return received, crc


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_listing(files: list[RemoteFile]) -> None:
    if not files:
        print("(no files on device)")
        return
    width = max(len(str(f.index)) for f in files)
    print(f"{'idx':>{width}}  {'size':>11}  path")
    print("-" * 78)
    for f in files:
        mb = f.size_bytes / (1024 * 1024)
        print(f"{f.index:>{width}d}  {f.size_bytes:>9d} B ({mb:5.1f} MB)  {f.path}")


def prompt_select(files: list[RemoteFile]) -> list[RemoteFile]:
    """Interactively ask the user which file(s) to download."""
    while True:
        ans = input("\nWhich file to download (index, comma-separated indices, "
                    "'all', or 'q' to quit)? ").strip().lower()
        if ans in ("q", "quit", "exit"):
            return []
        if ans == "all":
            return list(files)
        try:
            wanted = []
            for tok in ans.replace(",", " ").split():
                wanted.append(int(tok))
            picked = [f for f in files if f.index in wanted]
            missing = [i for i in wanted if i not in {f.index for f in files}]
            if missing:
                print(f"   (warning) index/indices not on device: {missing}")
            if picked:
                return picked
        except ValueError:
            pass
        print("   Please enter a number, a comma-separated list, 'all', or 'q'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OLA file-transfer-mode host. Lists and downloads .dat files "
                    "from an OLA running in file-transfer mode (2-press RESET).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--port", required=True,
                    help="Serial port the OLA is on (e.g. COM5 or /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=1_000_000,
                    help="Baud rate (default 1000000). On the OLA the USB connection "
                         "goes through a CH340 chip, so this is a real UART line rate "
                         "and must match the firmware's BAUD_RATE_USB. 1 Mbaud is the "
                         "tested-reliable rate on this hardware; pushing higher requires "
                         "both the firmware constant AND this default to be bumped, AND "
                         "the CH340 + host driver to agree.")
    ap.add_argument("--output-dir", type=Path, default=Path("."),
                    help="Where to save downloaded files (default: current dir)")
    ap.add_argument("--file-index", type=int, default=None,
                    help="Download this specific index non-interactively, then exit.")
    ap.add_argument("--all", action="store_true",
                    help="Download every file non-interactively, then exit.")
    ap.add_argument("--exit-after", action="store_true",
                    help="Send 'exit' to the device after the download(s), so the OLA "
                         "reboots back into normal logging mode.")
    args = ap.parse_args()

    print(f"Opening {args.port} at {args.baud} baud (DTR/RTS held LOW)...")
    # CRITICAL background: the OLA's USB connection goes through a CH340-style
    # USB-serial chip whose DTR pin is capacitively coupled to NRST. On
    # Windows, port enumeration toggles DTR briefly during open() *regardless*
    # of what pyserial sets — the OS-level driver does it before pyserial even
    # gets the handle. Setting dtr/rts LOW before open() reduces (but doesn't
    # always eliminate) the pulse.
    #
    # The robust workflow is therefore: open the port FIRST (accept that the
    # device may reboot once), then have the user enter file-transfer mode by
    # double-pressing RESET *after* we're connected. Once the port is held
    # open with DTR LOW, no further reset edges are generated and the device
    # stays in whatever mode it boots into.
    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = args.baud
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    try:
        ser.open()
    except serial.SerialException as e:
        print(f"ERROR: could not open {args.port}: {e}")
        return 1
    with ser:
        # Re-assert LOW after open in case the platform's driver clobbered it.
        ser.dtr = False
        ser.rts = False

        # Host-handshake protocol (no manual RESET press required):
        # The CH340 USB-serial chip on the OLA toggles DTR briefly when the
        # host opens the port (Windows always; some Linux distros too),
        # which capacitively couples through to NRST and reboots the
        # Apollo3. By the time we get here, the device is either rebooting
        # or partway through setup() — and the firmware will scan its
        # serial RX buffer for "OLA_ENTER_TRANSFER\n" after the multi-press
        # detection window. We spam that token at ~3 Hz during the boot
        # window so whatever bytes the CDC buffer manages to capture
        # contain the magic string. The firmware accepts and answers with
        # the file-transfer banner.
        print("Sending handshake (OLA_ENTER_TRANSFER spam) — waiting for OLA to enter "
              "file-transfer mode...")
        deadline = time.monotonic() + 20.0
        last_send = 0.0
        buf = bytearray()
        seen = False
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_send >= 0.3:
                ser.write(b"OLA_ENTER_TRANSFER\n")
                ser.flush()
                last_send = now
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)
                if b"FILE TRANSFER MODE" in buf or b"HANDSHAKE ACCEPTED" in buf:
                    seen = True
                    # Drain to next READY so we're synchronised.
                    try:
                        wait_for_ready(ser, timeout_s=8.0)
                    except TimeoutError:
                        pass
                    break
        if not seen:
            print()
            print("WARNING: did not see the file-transfer banner from the handshake.")
            print("Fallback: tap the RESET button on the OLA TWICE within ~2 s now.")
            print("(LED should start a slow cosine pulse.)")
            print("Waiting up to 60 s for the file-transfer banner...")
            deadline2 = time.monotonic() + 60.0
            while time.monotonic() < deadline2:
                chunk = ser.read(256)
                if chunk:
                    buf.extend(chunk)
                    if b"FILE TRANSFER MODE" in buf:
                        seen = True
                        try:
                            wait_for_ready(ser, timeout_s=5.0)
                        except TimeoutError:
                            pass
                        break
            if not seen:
                print("ERROR: device never entered file-transfer mode. Reset the OLA "
                      "and try again.")
                return 2
        print("Connected — file-transfer mode active.")

        # Drain any residual handshake spam + ERROR-echoes that the device
        # may have buffered before / during the transition into transfer
        # mode. The firmware also self-drains on entry, but we belt-and-
        # brace here to be sure nothing leftover poisons the first command.
        time.sleep(0.5)
        ser.reset_input_buffer()

        # List
        print("\nFetching file list...")
        files = cmd_list(ser)
        render_listing(files)

        # Pick which file(s)
        if args.all:
            picked = list(files)
        elif args.file_index is not None:
            picked = [f for f in files if f.index == args.file_index]
            if not picked:
                print(f"ERROR: index {args.file_index} not on device.")
                return 2
        else:
            picked = prompt_select(files)
            if not picked:
                print("Nothing to do.")
                return 0

        # Download
        for f in picked:
            local = args.output_dir / f.local_name
            print(f"\nDownloading idx={f.index} ({f.size_bytes} bytes) -> {local}")
            try:
                received, crc = cmd_get(ser, f.index, local)
            except Exception as e:
                print(f"   FAILED: {e}")
                continue
            print(f"   OK: {received} bytes, CRC32=0x{crc:08x}")

        if args.exit_after:
            print("\nTelling device to exit transfer mode...")
            send_command(ser, "exit")
            try:
                # Wait for the BYE line; the device reboots immediately after
                # so READY may never come back.
                read_line(ser, timeout_s=2.0)
            except TimeoutError:
                pass
            print("Device should be rebooting now.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
