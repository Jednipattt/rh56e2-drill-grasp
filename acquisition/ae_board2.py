"""Host side of the Board 2 acoustic-emission (AE) link.

Board 2 is a MOTION 2350 Pro (RP2350A) running MicroPython with the old AE
firmware (main.py + ae_*.py) on its flash. Nothing here ever writes, renames
or deletes a file on it. Two ways to stream:

    ram      paste ae_ram_sampler.py into the raw REPL (RAM only): 20 kHz/ch,
             64-sample frames (3.2 ms), u64 sample index, CRC-16, INFO.
    legacy   talk to the untouched main.py: ARM/START/STOP, 512-sample frames
             (25.6 ms), sum16 checksum, GP27/GP26 order (reordered here).

Either way frames come out as ch0 = GP26, ch1 = GP27, raw 12-bit counts, and
closing the link soft-resets the board so the untouched main.py runs again.

Boards are identified by USB serial number (rig_boards.json), never by COM
port, and the file list is checked read-only before anything is uploaded.

    py ae_board2.py --list            serials, ports and roles of plugged boards
    py ae_board2.py --info            connect (RAM sampler), print INFO, close
    py ae_board2.py --test 10         stream 10 s, print rate/drops/clock fit
    py ae_board2.py --test 10 --mode legacy
"""

import argparse
import binascii
import json
import struct
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("rig_boards.json")
RAM_SAMPLER_PATH = Path(__file__).with_name("ae_ram_sampler.py")
USB_VID_PID = (0x2E8A, 0x0005)          # MicroPython on RP2040/RP2350
MISSING = 0xFFFF                        # gap-fill value; 12-bit data is < 4096

FS_HZ = 20000
ADC_BITS = 12
# Conversion factors, with where each one comes from:
VREF_V = 3.3                    # Board 2 config.py: VREF = 3.3
FULL_SCALE_CODE = 4095          # 12-bit max code; old firmware convention is
                                # V = raw * VREF / max_code (preamp_mod.py:
                                # raw * VREF / 65535 for 16-bit read_u16)
DIVIDER_GAIN = (100e3 + 47e3) / 47e3    # ae_config.py: 0-10 V module output
                                        # "scaled to 0-3.2V via onboard
                                        # 100k/47k divider"
CHANNEL_PINS = (26, 27)

Frame = namedtuple("Frame", "seq first n flags samples stamp")
FLAG_OVERRUN, FLAG_FIFO, FLAG_FIRST = 1, 2, 4


class WrongBoardError(RuntimeError):
    """The port is not Board 2 (or its identity could not be confirmed)."""


# ---- board identification -------------------------------------------------
def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_boards(config=None):
    """[{port, serial, role}] for every MicroPython board plugged in now."""
    from serial.tools import list_ports
    config = config or load_config()
    roles = {(config.get(key) or {}).get("usb_serial"): key
             for key in ("board1", "board2")}
    roles.pop(None, None)
    return [{"port": p.device, "serial": p.serial_number,
             "role": roles.get(p.serial_number, "unknown")}
            for p in list_ports.comports()
            if (p.vid, p.pid) == USB_VID_PID]


def find_board(key, config=None):
    """COM port of board ``key`` ("board1"/"board2") by USB serial, or None."""
    config = config or load_config()
    serial_no = (config.get(key) or {}).get("usb_serial")
    if not serial_no:
        return None
    for board in list_boards(config):
        if board["serial"] == serial_no:
            return board["port"]
    return None


def port_serial(port):
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.device.upper() == str(port).upper():
            return p.serial_number
    return None


# ---- frame parsers ----------------------------------------------------------
class _Parser:
    """Shared buffer/resync/text handling. Bytes outside valid frames are
    scanned for text reply lines (INFO, STOPPED, ...)."""

    MAGIC = b""
    TEXT_PREFIXES = ()

    def __init__(self):
        self._buf = bytearray()
        self._text = bytearray()
        self.bad_frames = 0         # failed checksum / implausible header
        self.skipped_bytes = 0      # bytes discarded while resyncing
        self.lines = []             # text reply lines, oldest first

    def _junk(self, data):
        self._text.extend(data)
        while b"\n" in self._text:
            line, _, rest = self._text.partition(b"\n")
            self._text = bytearray(rest)
            text = line.decode("ascii", "replace").strip()
            if text.startswith(self.TEXT_PREFIXES):
                self.lines.append(text)
            elif text:
                self.skipped_bytes += len(line)
        if len(self._text) > 4096:
            self.skipped_bytes += len(self._text)
            self._text.clear()

    def feed(self, data, stamp):
        self._buf.extend(data)
        frames = []
        while True:
            idx = self._buf.find(self.MAGIC)
            if idx < 0:
                # hold back only a tail that could start a split magic
                keep = next((k for k in range(len(self.MAGIC) - 1, 0, -1)
                             if self._buf.endswith(self.MAGIC[:k])), 0)
                if len(self._buf) > keep:
                    self._junk(self._buf[:len(self._buf) - keep])
                    del self._buf[:len(self._buf) - keep]
                return frames
            if idx:
                self._junk(self._buf[:idx])
                del self._buf[:idx]
            result = self._parse_one(stamp)
            if result is None:          # incomplete: wait for more bytes
                return frames
            if result is False:         # bad header/checksum: skip a byte
                self.bad_frames += 1
                self.skipped_bytes += 1
                del self._buf[:1]
                continue
            frames.append(result)


class RamFrameParser(_Parser):
    """Frames from ae_ram_sampler.py (already ch0 = GP26, ch1 = GP27)."""

    MAGIC = b"\xA5\xE2"
    HDR = struct.Struct("<2sIQHH")
    TEXT_PREFIXES = ("INFO ", "STARTED", "STOPPED", "ERR", "Traceback",
                     "MemoryError", "OSError", "ValueError")
    MAX_N = 4096

    def _parse_one(self, stamp):
        buf = self._buf
        if len(buf) < self.HDR.size:
            return None
        _, seq, first, n, flags = self.HDR.unpack_from(buf)
        if not 0 < n <= self.MAX_N:
            return False
        end = self.HDR.size + 4 * n
        if len(buf) < end + 2:
            return None
        crc = buf[end] | buf[end + 1] << 8
        if binascii.crc_hqx(bytes(buf[2:end]), 0xFFFF) != crc:
            return False
        samples = np.frombuffer(bytes(buf[self.HDR.size:end]),
                                dtype="<u2").reshape(n, 2)
        del buf[:end + 2]
        return Frame(seq, first, n, flags, samples, stamp)


class LegacyFrameParser(_Parser):
    """Frames from the untouched main.py: magic AE 53 4D 50, seq u32,
    count u16, cumulative overrun u16, sum16 of payload; pairs are
    (GP27, GP26) and are reordered to (GP26, GP27) here. The sample index
    is seq * count (seq restarts at ARM, chunks are contiguous)."""

    MAGIC = b"\xAE\x53\x4D\x50"
    HDR = struct.Struct("<4sIHHH")
    TEXT_PREFIXES = ("ID ", "READY", "STOPPED", "ERROR", "DIAG", "Traceback")
    MAX_N = 100000

    def __init__(self):
        super().__init__()
        self.overruns = 0

    def _parse_one(self, stamp):
        buf = self._buf
        if len(buf) < self.HDR.size:
            return None
        _, seq, n, overrun, checksum = self.HDR.unpack_from(buf)
        if not 0 < n <= self.MAX_N:
            return False
        end = self.HDR.size + 4 * n
        if len(buf) < end:
            return None
        payload = np.frombuffer(bytes(buf[self.HDR.size:end]), dtype=np.uint8)
        if int(payload.sum(dtype=np.uint64)) & 0xFFFF != checksum:
            return False
        samples = payload.view("<u2").reshape(n, 2)[:, ::-1].copy()
        del buf[:end]
        flags = FLAG_FIRST if seq == 0 else 0
        if overrun != self.overruns & 0xFFFF:
            flags |= FLAG_OVERRUN
            self.overruns += (overrun - self.overruns) & 0xFFFF
        return Frame(seq, seq * n, n, flags, samples, stamp)


# ---- links ------------------------------------------------------------------
class Board2Link:
    """Open, verify and stream Board 2. Subclasses pick the protocol."""

    mode = None

    def __init__(self, port=None, config=None):
        import serial
        self.config = config or load_config()
        board2 = self.config["board2"]
        if not board2.get("usb_serial"):
            raise WrongBoardError("rig_boards.json has no board2 usb_serial")
        port = port or find_board("board2", self.config)
        if port is None:
            raise WrongBoardError(
                f"Board 2 (USB serial {board2['usb_serial']}) is not plugged in")
        serial_no = port_serial(port)
        if serial_no != board2["usb_serial"]:
            raise WrongBoardError(
                f"{port} has USB serial {serial_no}, not Board 2's "
                f"{board2['usb_serial']}: refusing to use it")
        if serial_no == (self.config.get("board1") or {}).get("usb_serial"):
            raise WrongBoardError(f"{port} is Board 1: refusing to use it")
        self.port, self.serial_no = port, serial_no
        self.ser = serial.Serial(port, 115200, timeout=0.2, write_timeout=2.0)
        self.info = {}
        self.files = []
        self.streaming = False
        self.parser = None
        try:
            self._enter_raw_repl()
            self.files = self._verify_files()
        except Exception:
            self.ser.close()
            raise

    # raw REPL helpers (read-only use: listing files)
    def _read_until(self, ending, timeout=5.0):
        deadline = time.time() + timeout
        data = bytearray()
        while time.time() < deadline:
            chunk = self.ser.read(max(1, self.ser.in_waiting))
            if chunk:
                data.extend(chunk)
                if data.endswith(ending):
                    return bytes(data)
        raise TimeoutError(f"Board 2 did not answer (got {bytes(data[-80:])!r})")

    def _enter_raw_repl(self):
        # Ctrl-C twice stops main.py (its finally: stops ADC/DMA) or an old
        # RAM sampler; Ctrl-A enters the raw REPL. Nothing touches flash.
        self.ser.write(b"\r\x03\x03")
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\x01")
        reply = self._read_until(b"raw REPL; CTRL-B to exit\r\n>")
        if b"raw REPL" not in reply:
            raise RuntimeError("could not enter the MicroPython raw REPL")

    def _exec(self, code, timeout=5.0):
        self.ser.write(code.encode() + b"\x04")
        reply = self._read_until(b"\x04>", timeout)
        if not reply.startswith(b"OK"):
            raise RuntimeError(f"Board 2 rejected the command: {reply!r}")
        out, _, err = reply[2:].partition(b"\x04")
        if err.strip(b"\x04>").strip():
            raise RuntimeError(err.decode(errors="replace").strip())
        return out.decode(errors="replace")

    def _verify_files(self):
        out = self._exec("import os\nprint(repr(sorted(os.listdir('/'))))")
        files = json.loads(out.strip().replace("'", '"'))
        board2 = self.config["board2"]
        missing = [f for f in board2.get("required_files", []) if f not in files]
        forbidden = [f for f in board2.get("forbidden_files", []) if f in files]
        if missing or forbidden:
            raise WrongBoardError(
                f"{self.port} does not look like Board 2 (missing {missing}, "
                f"has {forbidden}): refusing to use it")
        return files

    def _restart_main(self):
        """Leave the raw REPL and soft-reset: the untouched main.py runs."""
        self.ser.write(b"\r\x03\x03")
        time.sleep(0.1)
        self.ser.write(b"\x02")             # friendly REPL
        time.sleep(0.1)
        self.ser.write(b"\x04")             # soft reset -> boot.py/main.py
        time.sleep(0.3)

    def read(self):
        """Frames that arrived since the last call, each stamped with the
        time.time() taken right after the read returned (the /time/timestamp
        clock). Blocks up to the serial timeout when nothing is pending."""
        data = self.ser.read(1)
        stamp = time.time()
        if not data:
            return []
        waiting = self.ser.in_waiting
        if waiting:
            data += self.ser.read(waiting)
            stamp = time.time()
        return self.parser.feed(data, stamp)

    def close(self):
        try:
            if self.ser.is_open:
                self._shutdown()
        except Exception:
            pass
        finally:
            try:
                self.ser.close()
            except Exception:
                pass


class RamSamplerLink(Board2Link):
    mode = "ram_sampler"

    def __init__(self, port=None, config=None):
        super().__init__(port, config)
        self.parser = RamFrameParser()
        try:
            source = RAM_SAMPLER_PATH.read_bytes()
            for i in range(0, len(source), 256):     # plain raw REPL paste
                self.ser.write(source[i:i + 256])
                time.sleep(0.01)
            self.ser.write(b"\x04")
            reply = self._read_until(b"OK", timeout=5.0)
            self.parser.feed(reply.partition(b"OK")[2], time.time())
            self.info = self._wait_line("INFO ", timeout=5.0)
        except Exception:
            self.close()
            raise

    def _wait_line(self, prefix, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for i, line in enumerate(self.parser.lines):
                if line.startswith(prefix):
                    del self.parser.lines[:i + 1]
                    rest = line[len(prefix):].strip()
                    return json.loads(rest) if prefix == "INFO " else rest
                if line.startswith(("Traceback", "ERR", "MemoryError")):
                    raise RuntimeError(f"RAM sampler: {line}")
            self.parser.feed(self.ser.read(max(1, self.ser.in_waiting)),
                             time.time())
        raise TimeoutError(f"no {prefix.strip()} reply from the RAM sampler")

    def query_info(self):
        self.ser.write(b"INFO\n")
        self.info = self._wait_line("INFO ")
        return self.info

    def start(self):
        self.ser.write(b"START\n")
        self.streaming = True

    def stop(self):
        """Stop sampling; returns the board's STOPPED reply (frames, samples,
        overruns) or None. Frames still in flight are parsed and dropped."""
        if not self.streaming:
            return None
        self.ser.write(b"STOP\n")
        self.streaming = False
        try:
            return self._wait_line("STOPPED")
        except TimeoutError:
            return None

    def _shutdown(self):
        self.stop()
        self.ser.write(b"EXIT\n")
        try:
            self._read_until(b"\x04>", timeout=2.0)
        except TimeoutError:
            pass
        self._restart_main()


class LegacyStreamLink(Board2Link):
    mode = "legacy_stream"
    DEVICE_ID = "AE_SAMPLER_V1"

    def __init__(self, port=None, config=None):
        super().__init__(port, config)
        self.parser = LegacyFrameParser()
        try:
            self._restart_main()             # main.py from flash, unchanged
            self.ser.reset_input_buffer()
            reply = None
            for _ in range(10):
                self.ser.write(b"ID\n")
                try:
                    reply = self._wait_line("ID ", timeout=0.5)
                    break
                except TimeoutError:
                    continue
            if reply != self.DEVICE_ID:
                raise WrongBoardError(f"main.py answered ID {reply!r}, "
                                      f"expected {self.DEVICE_ID}")
            self.info = {"version": "main.py AE_SAMPLER_V1 (legacy stream)",
                         "fs_hz": FS_HZ, "adc_bits": ADC_BITS,
                         "pins": list(CHANNEL_PINS), "frame_samples": 512,
                         "checksum": "sum16 of payload",
                         "wire_order": "GP27, GP26 (reordered on the PC)"}
        except Exception:
            self.close()
            raise

    _wait_line = RamSamplerLink._wait_line

    def query_info(self):
        return self.info

    def start(self):
        self.ser.write(b"ARM\n")
        self._wait_line("READY")
        self.parser = LegacyFrameParser()
        self.ser.write(b"START\n")
        self.streaming = True

    def stop(self):
        if not self.streaming:
            return None
        self.ser.write(b"STOP\n")
        self.streaming = False
        try:
            reply = self._wait_line("STOPPED")
            self._wait_line("DIAG_END", timeout=2.0)
            return reply
        except TimeoutError:
            return None

    def _shutdown(self):
        # main.py is back in its idle loop after STOP: nothing else to undo.
        self.stop()


def open_link(mode="ram", port=None, config=None):
    cls = LegacyStreamLink if mode.startswith("legacy") else RamSamplerLink
    return cls(port, config)


# ---- clock mapping ----------------------------------------------------------
def fit_clock(end_index, arrival, fs_nominal=FS_HZ, window_s=1.0,
              iterations=5):
    """Fit arrival = a * end_index + b to the lower envelope of arrivals.

    USB and host delays only ever add latency, so per ``window_s`` window
    the frame arriving earliest relative to the current fit is kept, and
    the line is refit through those points. ``end_index`` is the index one
    past the frame's last sample (the frame is sendable once that sample is
    done), so the acquisition time of sample i is a * (i + 1) + b; the
    returned clock_b already includes that one-sample shift:
    t_pc(i) = clock_a * i + clock_b. A constant latency (the earliest
    possible USB delivery) remains unknown and is included in clock_b.
    """
    x = np.asarray(end_index, dtype=np.float64)
    y = np.asarray(arrival, dtype=np.float64)
    if len(x) < 2:
        return None
    x0, y0 = x[0], y[0]
    xr, yr = x - x0, y - y0
    a = 1.0 / fs_nominal
    b = float(np.min(yr - a * xr))
    window = np.floor(xr / (fs_nominal * window_s)).astype(np.int64)
    sel = np.arange(len(x))
    for _ in range(iterations):
        r = yr - (a * xr + b)
        order = np.lexsort((r, window))
        first = np.r_[True, window[order][1:] != window[order][:-1]]
        sel = order[first]
        if len(sel) < 2:
            break
        a, b = np.polyfit(xr[sel], yr[sel], 1)
    # pin the line to the envelope (no point earlier than the line)
    b += float(np.min((yr - (a * xr + b))[sel]))
    resid = yr - (a * xr + b)
    env = resid[sel]
    b_abs = y0 + b - a * x0
    return {
        "clock_a": float(a), "clock_b": float(b_abs + a),
        "fs_measured_hz": float(1.0 / a),
        "fs_error_ppm": float((1.0 / a / fs_nominal - 1.0) * 1e6),
        "fit_points": int(len(sel)), "frames": int(len(x)),
        "residual_rms_ms": float(np.sqrt(np.mean((env - env.mean()) ** 2))
                                 * 1e3),
        "envelope_max_ms": float(env.max() * 1e3),
        "latency_median_ms": float(np.median(resid) * 1e3),
        "latency_p99_ms": float(np.percentile(resid, 99) * 1e3),
        "latency_max_ms": float(resid.max() * 1e3),
        "window_s": float(window_s),
    }


# ---- command line -----------------------------------------------------------
def _test(mode, seconds):
    link = open_link(mode)
    print(f"{link.mode} on {link.port} (USB {link.serial_no})")
    print("INFO", json.dumps(link.query_info()))
    starts, stamps, seqs, total, gaps = [], [], [], 0, 0
    link.start()
    t_end = time.time() + seconds
    expected = None
    try:
        while time.time() < t_end:
            for f in link.read():
                if expected is not None and f.first != expected:
                    gaps += 1
                expected = f.first + f.n
                starts.append(f.first + f.n)
                stamps.append(f.stamp)
                seqs.append(f.seq)
                total += f.n
    finally:
        reply = link.stop()
        link.close()
    seqs = np.array(seqs)
    dropped = int(np.sum(np.diff(seqs) - 1)) if len(seqs) > 1 else 0
    fit = fit_clock(starts, stamps)
    dt = np.diff(stamps) * 1e3
    print(f"frames {len(seqs)}  samples {total}  dropped {dropped}  "
          f"index gaps {gaps}  bad {link.parser.bad_frames}  STOP: {reply}")
    if fit:
        print("clock fit:", json.dumps({k: round(v, 6) if isinstance(v, float)
                                        else v for k, v in fit.items()}))
    if len(dt):
        print("inter-arrival ms: p50 %.2f p90 %.2f p99 %.2f max %.1f  "
              "(%.0f%% < 0.5 ms)" % (*np.percentile(dt, [50, 90, 99]),
                                     dt.max(), 100 * np.mean(dt < 0.5)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--test", type=float, metavar="SECONDS")
    parser.add_argument("--mode", default="ram", choices=("ram", "legacy"))
    args = parser.parse_args(argv)
    if args.list or not (args.info or args.test):
        for board in list_boards():
            print(f"{board['port']:8} serial {board['serial']}  {board['role']}")
    if args.info:
        link = open_link(args.mode)
        try:
            print(json.dumps({"port": link.port, "serial": link.serial_no,
                              "mode": link.mode, "files": link.files,
                              "info": link.query_info()}, indent=2))
        finally:
            link.close()
    if args.test:
        _test(args.mode, args.test)


if __name__ == "__main__":
    sys.exit(main())
