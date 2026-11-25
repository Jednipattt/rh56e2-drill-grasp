"""RH56 web dashboard: hand control and HDF5 logging, with the linear guide, load
cell, webcam, acoustic emission and microphone synced to the same log.

Every piece of hardware is optional. The dashboard starts with whatever is
available and falls back per device:

    hand      real -> (connect fails / --demo) simulated hand
    guide     Pico on --guide-port -> (fails) disconnected panel
    webcam    --camera N -> (missing / no OpenCV) camera panel disabled
    mic       off unless --mic / page checkbox -> (no device) panel says why
    loadcell  HX711 on the guide Pico -> (--demo) simulated -> panel disabled
    ae        Board 2 by USB serial -> (absent / unplugged) panel says why

Run it:
    py Fingers_web_h5.py                 # try everything, fall back
    py Fingers_web_h5.py --demo          # no hardware at all
    py Fingers_web_h5.py --no-camera --guide-port ""

Sync with the log: `log start` also starts recording video into the log's
session folder, and `log stop` stops it. Both use time.time() -- the same
clock as /time/timestamp in the HDF5 file -- and every video frame's wall
time is written to <video>.frames.csv, so frame k of the video lines up with
the log sample whose timestamp is closest to frame_timestamp[k].
Manual control: `cam start [name]`, `cam stop`, `cam status`.

Microphone (optional): with --mic, `mic on` or the page's checkbox, `log
start` also records <session>/audio.wav (int16, --mic-rate, default 48 kHz)
and `log stop` ends it. audio.blocks.csv holds each 20 ms block's arrival
time on the same time.time() clock; audio.json says how to map samples.
`mic status`, `mic devices`, `mic device <index|name|default>`,
`mic start [name]`, `mic stop`, `mic off`.

Control mode: `mode` shows each finger's mode, `mode force` / `mode position`
switch all six, and `mode <finger> force|position` switches one. This drives
an undocumented register of the hand (1625, see CONTROL_MODE in H1.py).
The "Force control" panel also edits each finger's set force (FORCE_SET):
`force <finger> <g>` writes that one finger only, `forceset <g> x6` writes all
six at once.

Load cell: the HX711 (DT=GP17, SCK=GP16) hangs off the linear guide's Pico,
which samples it on its second core and buffers every reading, so guide moves
leave no gap. Whenever a log is open (`log start`, or any sequence that logs),
the samples go into the same .h5 under loadcell/ (timestamp, elapsed_s, raw,
force_g) on their own timeline, with the same clock as /time/timestamp. The
same Pico thread records the stage position at 50 Hz, also during moves, into
guide/ (timestamp, elapsed_s, position_mm, position_steps, homed).
`lc status`, `lc tare` (nothing on the cell) and `lc calibrate <grams>` (known
weight on the cell); the scale is kept in loadcell_calibration.json.

Acoustic emission: Board 2 (second MOTION 2350 Pro, AE-ENV24A10 envelopes on
GP26 = ch0 and GP27 = ch1) is found by USB serial in rig_boards.json, never by
COM port; Board 1 (guide) likewise with --guide-port auto. By default the PC
pastes ae_ram_sampler.py into Board 2's RAM (20 kHz x 2, 3.2 ms frames);
--ae-mode legacy reads the board's untouched main.py stream instead. Nothing
is ever written to Board 2, and disconnecting (or a reset) returns it to its
old firmware. A running log stores raw counts in ae/ (data, frame_first_index,
frame_pc_time, ...) with a clock fit t = clock_a * index + clock_b on the
/time/timestamp clock; the browser only gets 20 ms RMS. `ae status`,
`ae ports`, `ae info`, `ae connect [ram|legacy]`, `ae disconnect`.
"""


import argparse
import contextlib
import io
import json
import math
import mimetypes
import random
import shlex
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from Finghers_control_h5_v2 import (
    CANONICAL_FINGERS,
    FINGER_NAMES,
    FINGERS,
    TACTILE,
    tactile_label,
    FingerController,
    RH56Hand,
    RH56ModbusTCP,
    export_marked_data,
    export_tactile_video,
    expand_shortcut,
    load_shortcuts,
    plot_drill_calibration,
    plot_log,
    save_shortcuts,
    validate_shortcut_name,
)


class DemoTransport:
    """In-memory transport that behaves like the controller calls we use."""

    def __init__(self):
        self.hand = None
        self.speeds = [500] * 6
        self.force_limits = [1000] * 6

    def read_shorts(self, register):
        if register == "SPEED_SET":
            return list(self.speeds)
        if register == "FORCE_SET":
            return list(self.force_limits)
        if register == "POS_ACT":
            return [round((1000 - value) * 2) for value in self.hand.angles]
        raise ValueError(f"demo transport does not implement {register}")

    def read_tactile(self, region):
        _, _, rows, columns = TACTILE[region]
        finger = region.split("_", 1)[0]
        if finger == "palm":
            closure = sum((1000 - a) / 1000 for a in self.hand.angles[:5]) / 5
        else:
            index = FINGERS["little" if finger == "little" else finger]
            closure = (1000 - self.hand.angles[index]) / 1000

        # Contact begins around angle 700. The spatial hotspot makes the demo
        # heatmaps visibly respond while remaining stable between reads.
        pressure = max(0.0, (closure - 0.30) / 0.70)
        seed = sum(ord(char) for char in region)
        rng = random.Random(seed)
        center_r = (rows - 1) * (0.35 + 0.3 * rng.random())
        center_c = (columns - 1) * (0.35 + 0.3 * rng.random())
        spread = max(1.0, min(rows, columns) * 0.32)
        matrix = []
        for row in range(rows):
            values = []
            for column in range(columns):
                distance = ((row - center_r) ** 2
                            + (column - center_c) ** 2)
                hotspot = math.exp(-distance / (2 * spread ** 2))
                baseline = 8 + ((row * 17 + column * 11 + seed) % 10)
                values.append(round(baseline + pressure * 2600 * hotspot))
            matrix.append(values)
        return matrix

    def close(self):
        return None


class DemoHand:
    """Drop-in hand simulation for verifying the complete web interface."""

    def __init__(self):
        self.t = DemoTransport()
        self.t.hand = self
        self.angles = [1000] * 6
        self.status = [2] * 6
        self.errors = [0] * 6

    def _actual_forces(self):
        return [
            min(limit, round(max(0, 700 - angle) / 700 * 2200))
            for angle, limit in zip(self.angles, self.t.force_limits)
        ]

    def get_angles(self):
        return list(self.angles)

    def get_forces(self):
        return self._actual_forces()

    def get_currents(self):
        return [round(force * 0.22) for force in self._actual_forces()]

    def get_temperatures(self):
        return [26 + round(force / 900) for force in self._actual_forces()]

    def get_status(self, as_text=False):
        if as_text:
            return ["Reached position (demo)" for _ in self.status]
        return list(self.status)

    def get_errors(self):
        return list(self.errors)

    def set_angles(self, values):
        for index, value in enumerate(values):
            if value != -1:
                self.angles[index] = max(0, min(1000, int(value)))
                self.status[index] = 2
        return True

    def set_speeds(self, values):
        self.t.speeds = [int(value) for value in values]
        return True

    def set_forces(self, values):
        self.t.force_limits = [int(value) for value in values]
        return True

    def open_hand(self):
        return self.set_angles([1000] * 6)

    def close_hand(self):
        return self.set_angles([0] * 6)

    def wait_until_stopped(self):
        return True


class DemoLinearGuide:
    """Offline linear-guide model used with the web app's --demo flag."""

    mode = "demo"

    def __init__(self, travel_mm=120.0):
        self.position_mm = 0.0
        self.travel_mm = float(travel_mm)
        self.speed_max = 10.0
        self.homed = False
        self.enabled = False
        self.limit_hit = True

    def state(self):
        return {
            "mode": self.mode, "connected": True,
            "position_mm": round(self.position_mm, 3),
            "travel_mm": self.travel_mm, "speed_max": self.speed_max,
            "homed": self.homed, "enabled": self.enabled,
            "limit_hit": self.limit_hit,
        }

    def command(self, command, *values):
        command = command.upper()
        if command == "STATUS":
            return self.state()
        if command == "ENABLE":
            self.enabled = True
        elif command == "DISABLE":
            self.enabled = False
        elif command == "HOME":
            self.enabled = True
            self.position_mm = 0.0
            self.homed = True
            self.limit_hit = True
        elif command == "SPEED":
            speed = float(values[0])
            if speed <= 0:
                raise ValueError("speed must be greater than 0")
            self.speed_max = speed
        elif command in ("JOG", "MOVE"):
            if not self.homed and command == "MOVE":
                raise RuntimeError("home the guide before an absolute move")
            if not self.homed and abs(float(values[0])) > 5:
                raise RuntimeError("not homed — jog is limited to 5 mm")
            target = (self.position_mm + float(values[0])
                      if command == "JOG" else float(values[0]))
            self.position_mm = max(0.0, min(self.travel_mm, target))
            self.enabled = True
            self.limit_hit = self.position_mm <= 0
        else:
            raise ValueError(f"unknown guide command: {command}")
        return self.state()

    def close(self):
        return None


class SerialLinearGuide:
    """Access the unchanged Pico module through MicroPython's raw REPL."""

    mode = "real"

    def __init__(self, port, baudrate=115200, timeout=120.0):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "real guide control requires pyserial: pip install pyserial"
            ) from exc
        self.port = port
        self.timeout = float(timeout)
        self.serial = serial.Serial(port, baudrate=baudrate,
                                    timeout=0.25, write_timeout=2.0)
        self._lock = threading.Lock()
        self._enabled = False
        self._state = {
            "mode": self.mode, "connected": True, "port": port,
            "position_mm": None, "travel_mm": 120.0,
            "speed_max": None, "homed": False, "enabled": False,
            "limit_hit": False,
        }
        self._enter_raw_repl()
        self._exec(
            "import snapmarker_linear_module as _guide_module\n"
            "_web_guide = _guide_module.Stepper()"
        )
        self.command("STATUS")

    def _read_until(self, ending, timeout=5.0):
        deadline = time.time() + timeout
        data = bytearray()
        while time.time() < deadline:
            chunk = self.serial.read(max(1, self.serial.in_waiting))
            if chunk:
                data.extend(chunk)
                if data.endswith(ending):
                    return bytes(data)
        raise TimeoutError("MicroPython REPL did not respond")

    def _enter_raw_repl(self):
        time.sleep(0.3)
        self.serial.write(b"\r\x03\x03")
        time.sleep(0.2)
        self.serial.reset_input_buffer()
        self.serial.write(b"\r\x01")
        reply = self._read_until(b">", timeout=5.0)
        if b"raw REPL" not in reply:
            raise RuntimeError("could not enter the Pico MicroPython raw REPL")

    def _exec(self, code):
        self.serial.write(code.encode("utf-8") + b"\x04")
        response = self._read_until(b"\x04>", timeout=self.timeout)
        if not response.startswith(b"OK"):
            raise RuntimeError("Pico rejected the command")
        sections = response[2:].split(b"\x04")
        stdout = sections[0].decode("utf-8", errors="replace")
        stderr = (sections[1].decode("utf-8", errors="replace")
                  if len(sections) > 1 else "")
        if stderr.strip():
            raise RuntimeError(stderr.strip())
        return stdout

    def _read_state(self, preceding_code=""):
        code = preceding_code
        if code:
            code += "\n"
        code += (
            "print('__GUIDE__|{}|{}|{}|{}'.format("
            "_web_guide.get_mm(), int(_web_guide.homed), "
            "int(_web_guide.limit_hit()), _guide_module.SPEED_MAX))"
        )
        output = self._exec(code)
        markers = [line for line in output.splitlines()
                   if line.startswith("__GUIDE__|")]
        if not markers:
            raise RuntimeError("Pico returned no guide status")
        position, homed, limit_hit, speed = markers[-1].split("|")[1:]
        self._state.update({
            "position_mm": round(float(position), 4),
            "homed": bool(int(homed)), "enabled": self._enabled,
            "limit_hit": bool(int(limit_hit)), "speed_max": float(speed),
        })
        return self.state()

    def state(self):
        return dict(self._state)

    def command(self, command, *values):
        command = command.upper()
        with self._lock:
            if command == "STATUS":
                code = ""
            elif command == "ENABLE":
                code = "_web_guide.enable(True)"
                self._enabled = True
            elif command == "DISABLE":
                code = "_web_guide.enable(False)"
                self._enabled = False
            elif command == "HOME":
                code = "_web_guide.home()"
                self._enabled = True
            elif command == "JOG":
                speed = (f", {float(values[1])}"
                         if len(values) > 1 and values[1] is not None else "")
                code = f"_web_guide.move_mm({float(values[0])}{speed})"
                self._enabled = True
            elif command == "MOVE":
                speed = (f", {float(values[1])}"
                         if len(values) > 1 and values[1] is not None else "")
                code = f"_web_guide.move_to_mm({float(values[0])}{speed})"
                self._enabled = True
            elif command == "SPEED":
                speed = float(values[0])
                if speed <= 0:
                    raise ValueError("speed must be greater than 0")
                code = f"_guide_module.SPEED_MAX = {speed}"
            elif command == "CALIBRATE":
                code = "_web_guide.calibrate_overhead()"
            elif command == "BENCH":
                code = "_web_guide.bench()"
            else:
                raise ValueError(f"unknown guide command: {command}")
            return self._read_state(code)

    def close(self):
        if self.serial.is_open:
            try:
                with self._lock:
                    self._exec("_web_guide.enable(False)")
                    self.serial.write(b"\x02")
            except Exception:
                pass
            self.serial.close()


class DisconnectedLinearGuide:
    """Keeps the guide panel usable when no Pico serial port was selected."""

    mode = "disconnected"

    def state(self):
        return {"mode": self.mode, "connected": False,
                "message": "Start with --guide-port COMx to connect",
                "position_mm": None, "travel_mm": 120.0,
                "speed_max": None, "homed": False, "enabled": False,
                "limit_hit": False}

    def command(self, command, *values):
        raise RuntimeError("linear guide is not connected; use --guide-port")

    def close(self):
        return None


class RoutineAborted(Exception):
    """Raised inside a routine step when ``routine stop`` arrived first."""


def describe_step(step):
    """One-line label for a parsed routine step, shown in the console and UI."""
    kind, value = step
    return " ".join(value) if kind == "cmd" else f"{kind} {value:g} s"


class RoutineRunner:
    """Run a timed list of hand and guide commands on a background thread.

    Routines exist so one command can drive both machines on a schedule --
    "grasp with the middle finger, then move the guide five seconds later".
    The steps run off the request thread, so the dashboard keeps polling and
    the stop button stays live while a routine waits or a guide move blocks.
    """

    def __init__(self, step_runner):
        self._step_runner = step_runner
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._thread = None
        self._started = None
        self._state = self._idle_state()

    @staticmethod
    def _idle_state():
        return {"active": False, "index": 0, "total": 0, "current": None,
                "elapsed_s": 0.0, "message": "No routine has been run yet.",
                "steps": [], "log": []}

    @property
    def stopping(self):
        return self._abort.is_set()

    def status(self):
        with self._lock:
            state = dict(self._state)
            state["steps"] = list(state["steps"])
            state["log"] = list(state["log"])
            # Steps only publish elapsed time at their boundaries, so the
            # clock is recomputed here to keep ticking through a long wait.
            if state["active"] and self._started is not None:
                state["elapsed_s"] = round(time.monotonic() - self._started, 2)
        return state

    def _update(self, **changes):
        with self._lock:
            self._state.update(changes)

    def _record(self, index, elapsed, text, error=False):
        with self._lock:
            self._state["log"].append({"step": index,
                                       "elapsed_s": round(elapsed, 2),
                                       "text": text, "error": error})

    def start(self, steps):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "a routine is already running -- send 'routine stop' first"
                )
            self._abort.clear()
            self._started = time.monotonic()
            self._state = self._idle_state()
            self._state.update({
                "active": True, "total": len(steps),
                "message": "Routine running.",
                "steps": [describe_step(step) for step in steps],
            })
            self._thread = threading.Thread(target=self._run, args=(steps,),
                                            name="routine", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self):
        """Ask the routine to end. The step in progress still finishes.

        Nothing is joined here: a real guide move owns the serial link until
        the Pico replies, so joining would freeze the request thread with it.
        """
        self._abort.set()
        running = self._thread is not None and self._thread.is_alive()
        return {
            "stopping": running,
            "message": ("Stop requested; the step in progress finishes first."
                        if running else "No routine is running."),
            "routine": self.status(),
        }

    def _sleep(self, seconds):
        """Interruptible pause. True means ``routine stop`` arrived."""
        if seconds <= 0:
            return self._abort.is_set()
        return self._abort.wait(seconds)

    def _run(self, steps):
        started = self._started
        outcome = "Routine finished."
        for index, step in enumerate(steps, 1):
            kind, value = step
            self._update(index=index, current=describe_step(step),
                         elapsed_s=round(time.monotonic() - started, 2))
            if kind in ("wait", "at"):
                pause = (value if kind == "wait"
                         else value - (time.monotonic() - started))
                if self._sleep(pause):
                    outcome = "Routine stopped."
                    break
                self._record(index, time.monotonic() - started,
                             f"waited {max(0.0, pause):.2f} s" if kind == "wait"
                             else f"reached t = {value:g} s")
                continue
            try:
                printed = self._step_runner(value)
            except RoutineAborted:
                outcome = "Routine stopped."
                break
            except Exception as exc:
                outcome = f"step {index} ({describe_step(step)}) failed: {exc}"
                self._record(index, time.monotonic() - started, str(exc),
                             error=True)
                break
            self._record(index, time.monotonic() - started,
                         printed or f"{describe_step(step)} ok")
        self._update(active=False, current=None, message=outcome,
                     elapsed_s=round(time.monotonic() - started, 2))


def controller_state(fc, mode, guide=None):
    snap = fc._read_all()
    fingers = []
    for index, name in enumerate(FINGER_NAMES):
        fingers.append({
            "name": name,
            "angle": snap["angles"][index],
            "position": snap["positions"][index],
            "force": snap["forces"][index],
            "speed": snap["speeds"][index],
            "current": snap["currents"][index],
            "temperature": snap["temps"][index],
            "status": snap["status"][index],
            "error": snap["errors"][index],
        })
    state = {
        "mode": mode,
        "logging": fc.logger.active,
        "log_path": str(fc.logger.path) if fc.logger.path else None,
        "calibrated": fc.drill_calibration is not None,
        "fingers": fingers,
    }
    if guide is not None:
        state["guide"] = guide.state()
    return state


class CommandRunner:
    """Web-safe command adapter for the operations in the terminal script."""

    def __init__(self, fc, mode, guide=None):
        self.fc = fc
        self.mode = mode
        self.guide = guide or DisconnectedLinearGuide()
        self.lock = threading.Lock()
        self.routine = RoutineRunner(self._routine_step)
        try:
            self.shortcuts = load_shortcuts()
        except ValueError:
            self.shortcuts = {}

    def run(self, command):
        tokens = shlex.split(command)
        if not tokens:
            raise ValueError("enter a command")
        if (len(tokens) == 2 and tokens[0].lower() == "routine"
                and tokens[1].lower() in ("status", "stop")):
            # These skip the hardware lock so progress and the stop button
            # still answer while a slow step (a blocking guide move) holds it.
            result, printed = self._routine_command(tokens), ""
        else:
            result, printed = self._run_locked(tokens)
        return {
            "message": printed or self._message(result),
            "result": result,
            "state": controller_state(self.fc, self.mode, self.guide),
            "routine": self.routine.status(),
        }

    def _run_locked(self, tokens):
        with self.lock, contextlib.redirect_stdout(io.StringIO()) as output:
            result = self._execute(tokens)
            return result, output.getvalue().strip()

    def _routine_step(self, tokens):
        """Run one routine step under the lock console commands also take.

        The abort flag is re-checked here because the step may have been
        queued behind the lock while ``routine stop`` was being handled.
        """
        with self.lock, contextlib.redirect_stdout(io.StringIO()) as output:
            if self.routine.stopping:
                raise RoutineAborted
            result = self._execute(tokens)
            printed = output.getvalue().strip()
        # The routine log is one line per step, so the indented JSON that the
        # console shows would swamp it.
        if printed:
            return printed.replace("\n", " ")
        if isinstance(result, dict):
            return json.dumps(result, separators=(", ", "="))[1:-1].replace(
                '"', "")
        return self._message(result)

    @staticmethod
    def _message(result):
        if result is None:
            return "Command completed."
        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2)

    @staticmethod
    def _parse_guide(tokens):
        """Validate ``guide <action> [values]`` and return (action, values)."""
        if len(tokens) < 2:
            raise ValueError(
                "usage: guide <status|home|enable|disable|jog|move|speed>"
            )
        action = tokens[1].upper()
        aliases = {"MOVE_TO": "MOVE", "RELATIVE": "JOG"}
        action = aliases.get(action, action)
        values = tokens[2:]
        expected = {"STATUS": (0, 0), "ENABLE": (0, 0),
                    "DISABLE": (0, 0), "HOME": (0, 0),
                    "JOG": (1, 2), "MOVE": (1, 2),
                    "SPEED": (1, 1), "CALIBRATE": (0, 0),
                    "BENCH": (0, 0)}
        if action not in expected:
            raise ValueError(f"unknown guide action: {tokens[1]}")
        low, high = expected[action]
        if not low <= len(values) <= high:
            raise ValueError(f"wrong number of values for guide {action}")
        for value in values:
            try:
                float(value)
            except ValueError:
                raise ValueError(
                    f"guide {action} value must be a number: {value}"
                ) from None
        return action, values

    @staticmethod
    def _parse_routine(tokens):
        """Validate ``<step> [; <step> ...]`` and return the parsed steps.

        A step is ``("wait", seconds)`` -- pause after the previous step --
        ``("at", seconds)`` -- wait until that point on the routine clock --
        or ``("cmd", [tokens...])`` for any other console command.

        ``;`` always separates steps, spaced or not, so ``middle 0;wait 5``
        parses the same as ``middle 0 ; wait 5``.
        """
        flat = []
        for token in tokens:
            parts = token.split(";")
            for position, part in enumerate(parts):
                if position:
                    flat.append(";")
                if part:
                    flat.append(part)
        steps = []
        chunk = []
        for token in flat + [";"]:
            if token != ";":
                chunk.append(token)
                continue
            if not chunk:
                continue                    # tolerate empty or trailing ';'
            head = chunk[0].lower()
            if head in ("wait", "at"):
                if len(chunk) != 2:
                    raise ValueError(f"usage: {head} <seconds>")
                try:
                    seconds = float(chunk[1])
                except ValueError:
                    raise ValueError(
                        f"{head} needs a number of seconds: {chunk[1]}"
                    ) from None
                if seconds < 0:
                    raise ValueError(f"{head} seconds cannot be negative")
                steps.append((head, seconds))
            else:
                if head == "routine":
                    raise ValueError("a routine cannot start another routine")
                if head == "guide":
                    CommandRunner._parse_guide(chunk)   # fail before anything moves
                steps.append(("cmd", chunk))
            chunk = []
        if not steps:
            raise ValueError("a routine needs at least one step")
        return steps

    def _routine_command(self, tokens):
        action = tokens[1].lower() if len(tokens) > 1 else ""
        if action == "status":
            return self.routine.status()
        if action == "stop":
            return self.routine.stop()
        if action == "run":
            return self.routine.start(self._parse_routine(tokens[2:]))
        raise ValueError(
            "usage: routine run <step> [; <step> ...] | routine status | "
            "routine stop"
        )

    def _execute(self, tokens):
        cmd = tokens[0].lower()
        if cmd == "set":
            if len(tokens) < 3:
                raise ValueError("usage: set <name> <command...>")
            name = validate_shortcut_name(tokens[1])
            if name in ("guide", "routine"):
                raise ValueError(f"'{name}' is already a built-in command")
            head = tokens[2].lower()
            if head == "guide":
                self._parse_guide(tokens[2:])
            elif head == "routine":
                if len(tokens) < 4 or tokens[3].lower() != "run":
                    raise ValueError(
                        "save routines as: set <name> routine run <step> "
                        "[; <step> ...]"
                    )
                self._parse_routine(tokens[4:])
            self.shortcuts[name] = tokens[2:]
            save_shortcuts(self.shortcuts)
            return f"Saved shortcut {name}."
        if cmd == "unset":
            if len(tokens) != 2 or tokens[1].lower() not in self.shortcuts:
                raise ValueError("usage: unset <existing_name>")
            del self.shortcuts[tokens[1].lower()]
            save_shortcuts(self.shortcuts)
            return "Shortcut removed."
        if cmd == "shortcuts":
            return {name: " ".join(value)
                    for name, value in sorted(self.shortcuts.items())}

        tokens = expand_shortcut(tokens, self.shortcuts)
        cmd = tokens[0].lower()
        fc = self.fc
        if cmd == "guide":
            action, values = self._parse_guide(tokens)
            return self.guide.command(action, *values)
        if cmd == "routine":
            return self._routine_command(tokens)
        if cmd == "status":
            return {item["name"]: item["status"]
                    for item in controller_state(
                        fc, self.mode, self.guide)["fingers"]}
        if cmd == "angles":
            return dict(zip(FINGER_NAMES, fc.hand.get_angles()))
        if cmd == "forces":
            return dict(zip(FINGER_NAMES, fc.hand.get_forces()))
        if cmd == "speeds":
            return dict(zip(FINGER_NAMES,
                            fc.hand.t.read_shorts("SPEED_SET")))
        if cmd == "openall":
            fc.hand.open_hand()
        elif cmd == "closeall":
            fc.hand.close_hand()
        elif cmd == "wave":
            fc.wave_test(delay=0.15 if self.mode == "demo" else 0.6)
        elif cmd == "bend" and len(tokens) == 2:
            fc.bend(tokens[1])
        elif cmd == "open" and len(tokens) == 2:
            fc.open(tokens[1])
        elif cmd == "state" and len(tokens) == 2:
            return fc.finger_state(tokens[1])
        elif cmd == "all" and len(tokens) == 2:
            fc.hand.set_angles([int(tokens[1])] * 6)
        elif cmd == "speed" and len(tokens) == 3:
            fc.set_finger_speed(tokens[1], int(tokens[2]))
        elif cmd == "speedall" and len(tokens) == 2:
            value = int(tokens[1])
            if not 0 <= value <= 1000:
                raise ValueError("speed must be 0-1000")
            fc.hand.set_speeds([value] * 6)
        elif cmd == "force" and len(tokens) == 3:
            fc.set_finger_force(tokens[1], int(tokens[2]))
        elif cmd == "forceall" and len(tokens) == 2:
            value = int(tokens[1])
            if not 0 <= value <= 3000:
                raise ValueError("force must be 0-3000 g")
            fc.hand.set_forces([value] * 6)
        elif cmd == "group":
            args = tokens[1:]
            if len(args) < 2 or len(args) % 2:
                raise ValueError("usage: group <finger> <angle> [...]")
            fc.move_fingers({args[i]: int(args[i + 1])
                             for i in range(0, len(args), 2)})
        elif cmd == "seq":
            args = tokens[1:]
            if len(args) < 2 or len(args) % 2:
                raise ValueError("usage: seq <finger> <angle|delay ms> [...]")
            return fc.move_timed_sequence(
                [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
            )
        elif cmd == "logseq":
            args = tokens[1:]
            if len(args) < 2 or len(args) % 2:
                raise ValueError("usage: logseq <finger> <angle> [...]")
            return fc.move_sequence(
                [(args[i], int(args[i + 1]))
                 for i in range(0, len(args), 2)], log=True
            )
        elif cmd == "log":
            sub = tokens[1].lower() if len(tokens) > 1 else ""
            if sub == "start":
                return fc.start_log(tokens[2] if len(tokens) > 2 else None)
            if sub == "stop":
                log_path = fc.stop_log()
                if not log_path:
                    raise RuntimeError("no log has been started")
                plot_path = plot_log(log_path, show=False)
                tactile_plot = Path(log_path).with_name(
                    f"{Path(log_path).stem}_tactile.png"
                )
                return {
                    "log_path": log_path,
                    "plot_path": plot_path,
                    "tactile_plot": (str(tactile_plot)
                                     if tactile_plot.exists() else None),
                }
            raise ValueError("usage: log start [path] | log stop")
        elif cmd == "mark":
            return fc.mark_log(" ".join(tokens[1:]) or None)
        elif cmd == "plottactile":
            if len(tokens) > 2:
                raise ValueError("usage: plottactile [path]")
            return fc.plottactile(tokens[1] if len(tokens) == 2 else None)
        elif cmd == "plot":
            return plot_log(tokens[1] if len(tokens) > 1 else None,
                            show=False)
        elif cmd == "exportmarks":
            return export_marked_data(
                tokens[1] if len(tokens) > 1 else None,
                tokens[2] if len(tokens) > 2 else None,
            )
        elif cmd == "tactilevideo":
            return export_tactile_video(
                tokens[1] if len(tokens) > 1 else None,
                fps=float(tokens[2]) if len(tokens) > 2 else 5.0,
            )
        elif cmd == "drillsetcal" and len(tokens) in (2, 3):
            return fc.set_drill_calibration(
                int(tokens[1]), int(tokens[2]) if len(tokens) == 3 else None
            )
        elif cmd == "drill" and len(tokens) == 2:
            return {"target_angle": fc.set_drill_level(float(tokens[1]))}
        elif cmd == "drillrelease" and len(tokens) == 1:
            return {"target_angle": fc.release_drill_trigger()}
        elif cmd == "drillplot":
            return plot_drill_calibration(
                tokens[1] if len(tokens) > 1 else None, show=False
            )
        elif cmd == "drillcal":
            if len(tokens) < 2 or tokens[1] != "CONFIRM":
                raise ValueError(
                    "Safety: use 'drillcal CONFIRM [max_force_g]' only after "
                    "securing the drill and removing its bit."
                )
            maximum = int(tokens[2]) if len(tokens) > 2 else 3000
            return fc.calibrate_drill_trigger(max_force_g=maximum)
        elif cmd in FINGERS and len(tokens) in (2, 3):
            fc.move(cmd, int(tokens[1]),
                    force=int(tokens[2]) if len(tokens) == 3 else None)
        else:
            raise ValueError(f"unknown or incomplete command: {' '.join(tokens)}")
        return "Command completed."


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RH56 Hand Lab</title>
<style>
:root{--bg:#0b1220;--panel:#121d30;--panel2:#18263d;--text:#edf4ff;--muted:#91a4bd;--cyan:#31d7c8;--orange:#ffad5a;--red:#ff6b75;--line:#293b57}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#162a43 0,var(--bg) 42%);color:var(--text);font:14px Inter,Segoe UI,sans-serif}button,input,select{font:inherit}.wrap{max-width:1440px;margin:auto;padding:24px}.top{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:20px}.title h1{font-size:27px;margin:0 0 5px}.title p{color:var(--muted);margin:0}.badge{padding:8px 12px;border-radius:99px;background:#143d3b;color:var(--cyan);font-weight:700}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.panel{background:linear-gradient(145deg,var(--panel),#101a2b);border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 12px 28px #0003}.span8{grid-column:span 8}.span4{grid-column:span 4}.span12{grid-column:span 12}h2{font-size:16px;margin:0 0 14px}.toolbar,.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{border:1px solid #34516f;background:var(--panel2);color:var(--text);padding:9px 13px;border-radius:9px;cursor:pointer}button:hover{border-color:var(--cyan);transform:translateY(-1px)}button.primary{background:#16796f;border-color:#25aa9e}button.warn{background:#6d4220;border-color:#af7139}.fingers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.finger{border:1px solid var(--line);border-radius:12px;padding:12px;background:#0c1728}.finger-head{display:flex;justify-content:space-between;text-transform:capitalize;font-weight:700}.metric{color:var(--cyan)}input[type=range]{width:100%;accent-color:var(--cyan);margin:9px 0}.angle-row{display:grid;grid-template-columns:1fr 82px;gap:9px;align-items:center}.angle-row input[type=number]{width:100%;background:#091321;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:7px}.small{color:var(--muted);font-size:12px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px}.field label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.field input,.field select,.command input{width:100%;background:#091321;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:9px}.builders{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.builder{border:1px solid var(--line);border-radius:12px;padding:13px;background:#0c1728}.builder h3{font-size:14px;margin:0 0 9px}.builder input{width:100%;background:#091321;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:9px;margin-bottom:8px}textarea{width:100%;background:#091321;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:9px;font:13px ui-monospace,Consolas,monospace;resize:vertical}.routine-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:900px){.routine-grid{grid-template-columns:1fr}}.output{white-space:pre-wrap;background:#07101c;border:1px solid var(--line);border-radius:10px;min-height:100px;max-height:250px;overflow:auto;padding:12px;color:#bfe8df;margin-top:10px}.maps{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-top:12px}.map{background:#08121f;border-radius:8px;padding:6px;text-align:center;color:var(--muted);font-size:11px}.map canvas{width:100%;height:95px;image-rendering:pixelated}.notice{margin-top:10px;color:var(--muted)}a{color:var(--cyan)}.statusdot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--cyan);margin-right:6px}@media(max-width:900px){.span8,.span4{grid-column:span 12}.fingers{grid-template-columns:repeat(2,1fr)}.maps{grid-template-columns:repeat(3,1fr)}.builders{grid-template-columns:1fr}}@media(max-width:560px){.wrap{padding:14px}.fingers,.fields{grid-template-columns:1fr}.maps{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<div class="top"><div class="title"><h1>RH56 Hand Lab</h1><p>Control, observe, and capture tactile experiments.</p></div><div id="mode" class="badge">DEMO MODE</div></div>
<div class="grid">
<section class="panel span8"><h2>Live finger control</h2><div class="toolbar"><button class="primary" onclick="cmd('openall')">Open all</button><button class="warn" onclick="cmd('closeall')">Close all</button><input id="allAngle" type="number" min="0" max="1000" value="500" aria-label="Exact angle for all fingers" style="width:90px;background:#091321;border:1px solid var(--line);color:white;padding:9px;border-radius:8px"><button onclick="cmd('all '+$('allAngle').value)">Set all exact</button><button onclick="cmd('wave')">Wave test</button><button onclick="refresh()">Refresh</button></div><div id="fingers" class="fingers"></div></section>
<section class="panel span4"><h2>Experiment controls</h2><div class="fields"><div class="field"><label>Finger</label><select id="target"><option>little</option><option>ring</option><option>middle</option><option selected>index</option><option>thumb</option><option>thumb_rot</option></select></div><div class="field"><label>Exact value</label><input id="setting" type="number" min="0" max="3000" value="500"></div></div><div class="toolbar" style="margin-top:10px"><button onclick="setting('speed')">Set speed</button><button onclick="setting('force')">Set force</button></div><hr style="border:0;border-top:1px solid var(--line);margin:17px 0"><div class="field"><label>Drill trigger level (%)</label><div class="angle-row"><input id="drill" type="range" min="0" max="100" value="50" oninput="$('drillNumber').value=this.value"><input id="drillNumber" type="number" min="0" max="100" value="50" oninput="$('drill').value=this.value"></div></div><div class="toolbar"><button onclick="cmd('drill '+$('drillNumber').value)">Set level</button><button onclick="cmd('drillrelease')">Release</button></div><div class="fields" style="margin-top:12px"><div class="field"><label>Contact angle</label><input id="contact" type="number" min="0" max="1000" value="700"></div><div class="field"><label>Deepest angle</label><input id="deepest" type="number" min="0" max="1000" value="300"></div></div><button style="margin-top:10px" onclick="cmd('drillsetcal '+$('contact').value+' '+$('deepest').value)">Set drill calibration</button></section>
<section class="panel span4"><h2>Data logging</h2><div class="toolbar"><button class="primary" onclick="cmd('log start')">Start log</button><button onclick="stopLogging()">Stop + generate plots</button></div><div class="row" style="margin-top:11px"><input id="mark" placeholder="mark label" style="flex:1;background:#091321;border:1px solid var(--line);color:white;padding:9px;border-radius:8px"><button onclick="cmd('mark '+$('mark').value)">Mark now</button></div><p id="logState" class="notice"><span class="statusdot"></span>Not logging</p><div id="logFiles" class="notice"></div></section>
<section class="panel span8"><h2>Instant tactile status</h2><div class="toolbar"><button class="primary" onclick="captureTactile()">Capture HDF5 + PNG now</button><button onclick="loadTactile()">Preview current sensors</button></div><div id="capture" class="notice">No snapshot captured yet.</div><div id="maps" class="maps"></div></section>
<section class="panel span8"><h2>Sequences and shortcuts</h2><div class="builders"><div class="builder"><h3>Group movement</h3><input id="groupArgs" value="index 450 thumb 400" aria-label="Group finger and angle pairs"><div class="small">Enter finger/angle pairs. They move together.</div><button class="primary" style="margin-top:9px" onclick="runBuilder('group','groupArgs')">Run group</button></div><div class="builder"><h3>Sequential movement</h3><input id="seqArgs" value="index 800 delay 500 index 400" aria-label="Sequence actions"><div class="small">Use finger/angle pairs and delay/milliseconds.</div><button class="primary" style="margin-top:9px" onclick="runBuilder('seq','seqArgs')">Run sequence</button></div><div class="builder"><h3>Create or update control</h3><input id="shortcutName" placeholder="control name" aria-label="Shortcut name"><input id="shortcutCommand" placeholder="e.g. group index 450 thumb 400" aria-label="Shortcut command"><button class="primary" onclick="saveShortcut()">Save as web control</button></div></div><div style="margin-top:15px"><div class="row"><h3 style="margin:0;flex:1">Saved web controls</h3><button onclick="refreshShortcuts()">Refresh controls</button></div><div id="shortcutList" class="toolbar" style="margin-top:10px"><span class="small">No saved controls.</span></div></div></section>
<section class="panel span4 guide-panel"><h2>Snapmaker linear guide</h2><div id="guideState" class="notice">Loading guide status…</div><div class="toolbar" style="margin:12px 0"><button class="primary" onclick="guide('home')">Home</button><button onclick="guide('enable')">Enable</button><button onclick="guide('disable')">Disable</button></div><div class="fields"><div class="field"><label>Jog distance (mm)</label><input id="guideJog" type="number" step="0.1" value="5"></div><div class="field"><label>Motion speed (mm/s)</label><input id="guideSpeed" type="number" min="0.1" step="0.1" value="10"></div></div><div class="toolbar" style="margin-top:10px"><button onclick="guideJog(-1)">Jog −</button><button onclick="guideJog(1)">Jog +</button></div><div class="field" style="margin-top:12px"><label>Absolute target (mm)</label><input id="guideTarget" type="number" min="0" step="0.1" value="0"></div><div class="toolbar" style="margin-top:10px"><button class="primary" onclick="guideMove()">Move absolute</button><button onclick="guide('speed',$('guideSpeed').value)">Set max speed</button></div><div style="margin-top:15px;border-top:1px solid var(--line);padding-top:13px"><div class="row"><h3 style="margin:0;flex:1;font-size:14px">Saved guide controls</h3><button onclick="refreshShortcuts()">Refresh</button></div><div class="fields" style="margin-top:10px"><div class="field"><label>Control name</label><input id="guideShortcutName" placeholder="e.g. g_home_pos"></div><div class="field"><label>Guide command</label><input id="guideShortcutCommand" placeholder="e.g. move 50 10"></div></div><div class="small" style="margin-top:6px">Actions: home · enable · disable · jog &lt;mm&gt; [mm/s] · move &lt;mm&gt; [mm/s] · speed &lt;mm/s&gt;</div><div class="toolbar" style="margin-top:9px"><button class="primary" onclick="saveGuideShortcut()">Save guide control</button><button onclick="saveGuideShortcut('move '+$('guideTarget').value+' '+$('guideSpeed').value)" title="Save a move to the Absolute target at the Motion speed">Save current target</button></div><div id="guideShortcutList" class="toolbar" style="margin-top:10px"><span class="small">No saved guide controls.</span></div></div><details style="margin-top:13px"><summary class="small" style="cursor:pointer">Diagnostics</summary><div class="toolbar" style="margin-top:9px"><button onclick="guideDiagnostic('calibrate')">Calibrate loop overhead</button><button class="warn" onclick="guideDiagnostic('bench')">Run benchmark</button></div></details></section>
<section class="panel span12 routine-panel"><h2>Timed routine &mdash; hand + linear guide</h2><div class="routine-grid"><div class="builder"><h3>Steps (one per line)</h3><textarea id="routineSteps" rows="8" spellcheck="false" aria-label="Routine steps">middle 0
wait 5
guide move 106 10</textarea><div class="small" style="margin-top:8px">Any console command works as a step. <b>wait &lt;s&gt;</b> pauses after the previous step; <b>at &lt;s&gt;</b> waits until that many seconds after the routine started. Finger moves are commanded and the clock keeps running while they travel; a guide move finishes before the next step begins. Lines starting with # are ignored.</div><div class="toolbar" style="margin-top:10px"><button class="primary" onclick="runRoutine()">Run routine</button><button class="warn" onclick="stopRoutine()">Stop</button><button onclick="cmd('routine status')">Status to console</button></div></div><div class="builder"><h3>Progress</h3><div id="routineState" class="notice">No routine has been run yet.</div><div id="routineLog" class="output" style="min-height:96px;max-height:180px">No steps have run yet.</div><div class="fields" style="margin-top:12px"><div class="field"><label>Save these steps as a control</label><input id="routineName" placeholder="e.g. grasp_then_advance"></div><div class="field" style="display:flex;align-items:flex-end"><button class="primary" style="width:100%" onclick="saveRoutine()">Save routine control</button></div></div><div style="margin-top:15px;border-top:1px solid var(--line);padding-top:13px"><div class="row"><h3 style="margin:0;flex:1;font-size:14px">Saved routine controls</h3><button onclick="refreshShortcuts()">Refresh</button></div><div id="routineShortcutList" class="toolbar" style="margin-top:10px"><span class="small">No saved routine controls.</span></div></div></div></div></section>
<section class="panel span12"><h2>Command console</h2><div class="command row"><input id="command" placeholder="Examples: group index 450 thumb 400  |  seq index 800 delay 500 index 400" onkeydown="if(event.key==='Enter')runConsole()"><button class="primary" onclick="runConsole()">Run</button></div><div id="output" class="output">Dashboard ready. Demo mode does not connect to hardware.</div></section>
</div></div><script>
const $=id=>document.getElementById(id);let state=null;
async function api(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data}
async function refresh(){try{state=await api('/api/state');const active=document.activeElement;if(!active||!active.closest||(!active.closest('.finger')&&!active.closest('.guide-panel')&&!active.closest('.routine-panel')))renderState(state)}catch(e){show(e.message,true)}}
function renderState(s){$('mode').textContent=s.mode==='demo'?'DEMO MODE · NO ROBOT':'REAL HAND';$('mode').style.background=s.mode==='demo'?'#44371c':'#143d3b';$('fingers').innerHTML=s.fingers.map(f=>`<div class="finger"><div class="finger-head"><span>${f.name.replace('_',' ')}</span><span class="metric">actual ${f.angle}</span></div><div class="angle-row"><input type="range" min="0" max="1000" value="${f.angle}" id="r-${f.name}" oninput="document.getElementById('n-${f.name}').value=this.value"><input type="number" min="0" max="1000" value="${f.angle}" id="n-${f.name}" oninput="document.getElementById('r-${f.name}').value=this.value"></div><div class="row"><button onclick="move('${f.name}')">Move exact</button><span class="small">${f.force} g · ${f.current} mA · ${f.temperature}°C</span></div></div>`).join('');$('logState').innerHTML=`<span class="statusdot" style="background:${s.logging?'var(--orange)':'var(--cyan)'}"></span>${s.logging?'Logging: '+s.log_path:'Not logging'}`;renderGuide(s.guide)}
function renderGuide(g){if(!g)return;const color=g.connected?'var(--cyan)':'var(--red)',position=g.position_mm==null?'unknown':Number(g.position_mm).toFixed(2)+' mm';$('guideState').innerHTML=`<span class="statusdot" style="background:${color}"></span>${g.connected?(g.mode==='demo'?'Demo guide':'Connected '+(g.port||'')):(g.message||'Disconnected')}<br><span class="metric">${position}</span> · ${g.homed?'homed':'not homed'} · ${g.enabled?'enabled':'disabled'} · limit ${g.limit_hit?'active':'clear'} · max ${g.speed_max==null?'—':g.speed_max+' mm/s'}`}
async function cmd(command){show('Running: '+command);try{const data=await api('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command})});show(data.message||'Completed');state=data.state;renderState(state);renderRoutine(data.routine);return data.result}catch(e){show(e.message,true);throw e}}
function show(text,error=false){$('output').textContent=(error?'ERROR: ':'')+text+'\n\n'+$('output').textContent}
function move(name){cmd(name+' '+$('n-'+name).value)}function setting(kind){cmd(kind+' '+$('target').value+' '+$('setting').value)}function runConsole(){const value=$('command').value.trim();if(value){cmd(value);$('command').value=''}}
function runBuilder(command,inputId){const args=$(inputId).value.trim();if(args)cmd(command+' '+args)}
function routineLines(){return $('routineSteps').value.split('\n').map(line=>line.trim()).filter(line=>line&&!line.startsWith('#'))}
function runRoutine(){const steps=routineLines();if(!steps.length){show('Add at least one routine step.',true);return}cmd('routine run '+steps.join(' ; '))}
function stopRoutine(){cmd('routine stop')}
async function saveRoutine(){const name=$('routineName').value.trim(),steps=routineLines();if(!name||!steps.length){show('Routine name and at least one step are required.',true);return}try{await cmd('set '+name+' routine run '+steps.join(' ; '));$('routineName').value='';await refreshShortcuts()}catch(e){}}
function renderRoutine(r){if(!r)return;const box=$('routineState');box.replaceChildren();const dot=document.createElement('span');dot.className='statusdot';dot.style.background=r.active?'var(--orange)':'var(--cyan)';const message=document.createElement('span');message.textContent=r.message;const meta=document.createElement('span');meta.className='small';meta.textContent=(r.active?'step '+r.index+' / '+r.total:(r.total?r.total+' steps':'idle'))+' \u00b7 '+Number(r.elapsed_s||0).toFixed(1)+' s'+(r.current?' \u00b7 now: '+r.current:'');box.append(dot,message,document.createElement('br'),meta);$('routineLog').textContent=r.log.length?r.log.map(e=>'['+Number(e.elapsed_s).toFixed(2)+'s] '+e.step+'. '+(e.error?'ERROR ':'')+e.text).join('\n'):'No steps have run yet.'}
async function refreshRoutine(){try{renderRoutine(await api('/api/routine'))}catch(e){}}
async function guide(action,...values){return cmd('guide '+action+(values.length?' '+values.join(' '):''))}
function guideJog(sign){const distance=Math.abs(Number($('guideJog').value))*sign;guide('jog',distance,$('guideSpeed').value)}
function guideMove(){guide('move',$('guideTarget').value,$('guideSpeed').value)}
function guideDiagnostic(action){const warning=action==='bench'?'This homes and repeatedly moves the full guide. Continue?':'This disables the driver and pulses the STEP output for calibration. Continue?';if(confirm(warning))guide(action)}
async function stopLogging(){try{const result=await cmd('log stop'),links=[['HDF5 log',result.log_path],['Parameter plot',result.plot_path],['Tactile heatmap',result.tactile_plot]].filter(([,path])=>path);$('logFiles').innerHTML='Generated: '+links.map(([label,path])=>`<a href="/files?path=${encodeURIComponent(path)}" target="_blank">${label}</a>`).join(' · ')}catch(e){}}
async function saveShortcut(){const name=$('shortcutName').value.trim(),body=$('shortcutCommand').value.trim();if(!name||!body){show('Control name and command are required.',true);return}try{await cmd('set '+name+' '+body);$('shortcutName').value='';$('shortcutCommand').value='';await refreshShortcuts()}catch(e){}}
async function removeShortcut(name){try{await cmd('unset '+name);await refreshShortcuts()}catch(e){}}
async function saveGuideShortcut(preset){const name=$('guideShortcutName').value.trim();let body=(preset||$('guideShortcutCommand').value).trim().replace(/^guide\s+/i,'');if(!name||!body){show('Guide control name and command are required.',true);return}try{await cmd('set '+name+' guide '+body);$('guideShortcutName').value='';$('guideShortcutCommand').value='';await refreshShortcuts()}catch(e){}}
const isGuideCommand=command=>/^guide(\s|$)/i.test(command);
const isRoutineCommand=command=>/^routine(\s|$)/i.test(command);
async function refreshShortcuts(){try{const shortcuts=await api('/api/shortcuts'),entries=Object.entries(shortcuts);renderShortcutList($('shortcutList'),entries.filter(([,c])=>!isGuideCommand(c)&&!isRoutineCommand(c)),'No saved controls.',false);renderShortcutList($('guideShortcutList'),entries.filter(([,c])=>isGuideCommand(c)),'No saved guide controls.',true);renderShortcutList($('routineShortcutList'),entries.filter(([,c])=>isRoutineCommand(c)),'No saved routine controls.',false)}catch(e){show(e.message,true)}}
function renderShortcutList(list,entries,emptyText,isGuide){list.replaceChildren();if(!entries.length){const empty=document.createElement('span');empty.className='small';empty.textContent=emptyText;list.appendChild(empty);return}entries.forEach(([name,command])=>{const box=document.createElement('div');box.className='row';box.style.cssText='border:1px solid var(--line);border-radius:10px;padding:8px;background:#0c1728';const run=document.createElement('button');run.className='primary';run.textContent=name;run.title='Run: '+command;run.onclick=()=>cmd(name);const detail=document.createElement('span');detail.className='small';detail.textContent=isGuide?command.replace(/^guide\s+/i,''):command;const remove=document.createElement('button');remove.textContent='Delete';remove.onclick=()=>removeShortcut(name);box.append(run,detail,remove);list.appendChild(box)})}
async function loadTactile(){try{const data=await api('/api/tactile');renderMaps(data.tactile)}catch(e){show(e.message,true)}}
function renderMaps(tactile){const entries=Object.entries(tactile);let maximum=1;entries.forEach(([,m])=>m.forEach(r=>r.forEach(v=>maximum=Math.max(maximum,v))));$('maps').innerHTML=entries.map(([name])=>`<div class="map"><canvas id="map-${name}"></canvas><div>${name.replace('_',' ')}</div></div>`).join('');entries.forEach(([name,m])=>{const c=$('map-'+name),x=c.getContext('2d'),rows=m.length,cols=m[0].length;c.width=cols;c.height=rows;const img=x.createImageData(cols,rows);m.forEach((row,r)=>row.forEach((value,col)=>{const t=Math.min(1,value/maximum),i=(r*cols+col)*4;img.data[i]=Math.round(255*Math.min(1,t*2.2));img.data[i+1]=Math.round(190*Math.max(0,(t-.25)*1.6));img.data[i+2]=Math.round(80*Math.max(0,(t-.72)*3.5));img.data[i+3]=255}));x.putImageData(img,0,0)})}
async function captureTactile(){try{const result=await cmd('plottactile');const h5='/files?path='+encodeURIComponent(result.h5_path),png='/files?path='+encodeURIComponent(result.png_path);$('capture').innerHTML=`Captured ${result.timestamp}: <a href="${h5}" target="_blank">HDF5</a> · <a href="${png}" target="_blank">PNG</a>`;await loadTactile()}catch(e){}}
refresh();loadTactile();refreshShortcuts();refreshRoutine();setInterval(refresh,2500);setInterval(refreshRoutine,1000);
</script></body></html>"""


def make_handler(runner, workspace):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"  [web] {self.address_string()} - {fmt % args}")

        def json_response(self, data, status=200):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    payload = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif parsed.path == "/api/state":
                    self.json_response(controller_state(
                        runner.fc, runner.mode, runner.guide
                    ))
                elif parsed.path == "/api/guide":
                    self.json_response(runner.guide.state())
                elif parsed.path == "/api/routine":
                    self.json_response(runner.routine.status())
                elif parsed.path == "/api/tactile":
                    self.json_response({
                        "timestamp": time.time(),
                        "tactile": {
                            tactile_label(region): matrix for region, matrix
                            in runner.fc._read_tactile_all().items()
                        },
                    })
                elif parsed.path == "/api/shortcuts":
                    self.json_response({
                        name: " ".join(command)
                        for name, command in sorted(runner.shortcuts.items())
                    })
                elif parsed.path == "/files":
                    self.serve_file(parse_qs(parsed.query).get("path", [""])[0])
                else:
                    self.json_response({"error": "not found"}, 404)
            except Exception as exc:
                self.json_response({"error": str(exc)}, 400)

        def do_POST(self):
            if urlparse(self.path).path != "/api/command":
                self.json_response({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                self.json_response(runner.run(str(body.get("command", ""))))
            except Exception as exc:
                self.json_response({"error": str(exc)}, 400)

        def serve_file(self, requested):
            path = Path(requested)
            if not path.is_absolute():
                path = workspace / path
            path = path.resolve()
            if workspace != path and workspace not in path.parents:
                raise PermissionError("file is outside the workspace")
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",
                             mimetypes.guess_type(path.name)[0]
                             or "application/octet-stream")
            self.send_header("Content-Disposition",
                             f"inline; filename={quote(path.name)}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


# ----------------------------------------------------------------------
# Webcam, microphone, load cell, acoustic emission and the force-control panel
# ----------------------------------------------------------------------

import argparse
import csv
import json
import math
import random
import sys
import threading
import time
import wave
import webbrowser
from collections import deque
from http.server import ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from urllib.parse import parse_qs, urlparse


try:
    import cv2
except ImportError:              # camera features simply switch off
    cv2 = None

try:
    import sounddevice as sd
except (ImportError, OSError):   # microphone features simply switch off
    sd = None

CONTROL_MODES = ("position", "force")       # index = value in the register


class ModeDemoHand(DemoHand):
    """Simulated hand that also remembers the control-mode register."""

    def __init__(self):
        super().__init__()
        self.control_modes = [0] * 6

    def get_control_modes(self):
        return list(self.control_modes)

    def set_control_modes(self, values):
        self.control_modes = [1 if value else 0 for value in values]
        return True

    def set_force(self, index, value):
        self.t.force_limits[index] = int(value)
        return True


def read_control_modes(fc):
    """Control mode name per DOF, or None if the hand does not report it."""
    if getattr(fc, "control_mode_unreadable", False):
        return None
    try:
        with fc.io_lock:
            raw = fc.hand.get_control_modes()
    except Exception:
        # The register is undocumented, so a hand may not answer for it. Stop
        # asking, or every state poll would wait on the same failed read.
        fc.control_mode_unreadable = True
        return None
    return [CONTROL_MODES[1 if value else 0] for value in raw]


_base_controller_state = controller_state   # the dashboard's own version, defined above


def controller_state(fc, mode, guide=None):
    """The base dashboard state plus each finger's control mode and set force."""
    state = _base_controller_state(fc, mode, guide)
    modes = read_control_modes(fc)
    with fc.io_lock:
        force_set = fc.hand.t.read_shorts("FORCE_SET")
    for index, finger in enumerate(state["fingers"]):
        finger["control_mode"] = modes[index] if modes else None
        finger["force_set"] = force_set[index]
    return state




class Webcam:
    """Optional webcam: live MJPEG preview plus timestamped video recording.

    Never raises on a missing device -- ``available`` is just False and the
    rest of the app keeps working.
    """

    def __init__(self, index=0, width=640, height=480, fps=30.0):
        self.index, self.width, self.height = index, width, height
        self.fps = float(fps)
        self.available = False
        self.message = "camera not started"
        self._cap = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._jpeg = None
        self._writer = None
        self._frames_csv = None
        self._frames_file = None
        self._frame_no = 0
        self.video_path = None
        self.record_started = None

    def open(self):
        if cv2 is None:
            self.message = "OpenCV missing: pip install opencv-python"
            return False
        try:
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                self.message = f"no webcam at index {self.index}"
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                self.message = "webcam opened but returned no frames"
                return False
        except Exception as exc:
            self.message = f"webcam error: {exc}"
            return False
        self._cap = cap
        self.height, self.width = frame.shape[:2]
        self.available = True
        self.message = f"webcam {self.index} ({self.width}x{self.height})"
        self._thread = threading.Thread(target=self._loop, name="webcam",
                                        daemon=True)
        self._thread.start()
        return True

    @property
    def recording(self):
        return self._writer is not None

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            stamp = time.time()          # same clock as the HDF5 log
            if not ok:
                time.sleep(0.05)
                continue
            good, buf = cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                if good:
                    self._jpeg = buf.tobytes()
                if self._writer is not None:
                    self._writer.write(frame)
                    self._frames_csv.writerow(
                        [self._frame_no, f"{stamp:.6f}",
                         f"{stamp - self.record_started:.6f}"])
                    self._frame_no += 1

    def jpeg(self):
        with self._lock:
            return self._jpeg

    def start_recording(self, path):
        if not self.available:
            raise RuntimeError(self.message)
        if self.recording:
            raise RuntimeError(f"already recording to {self.video_path}")
        path = Path(path).with_suffix(".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot write video: {path}")
        frames_file = open(path.with_suffix(".frames.csv"), "w",
                           newline="", encoding="utf-8")
        csv_writer = csv.writer(frames_file)
        csv_writer.writerow(["frame", "timestamp", "elapsed_s"])
        with self._lock:
            self._frame_no = 0
            self.record_started = time.time()
            self._frames_file, self._frames_csv = frames_file, csv_writer
            self.video_path = path
            self._writer = writer
        path.with_suffix(".json").write_text(json.dumps({
            "video": path.name, "frames_csv": path.with_suffix(".frames.csv").name,
            "start_timestamp": self.record_started, "fps": self.fps,
            "width": self.width, "height": self.height,
            "clock": "time.time() (same as HDF5 /time/timestamp)",
        }, indent=2), encoding="utf-8")
        return str(path)

    def stop_recording(self):
        with self._lock:
            writer, frames = self._writer, self._frames_file
            self._writer = self._frames_csv = self._frames_file = None
            count = self._frame_no
        if writer is None:
            return None
        writer.release()
        frames.close()
        return {"video_path": str(self.video_path), "frames": count}

    def state(self):
        return {"available": self.available, "message": self.message,
                "recording": self.recording,
                "video_path": str(self.video_path) if self.recording else None}

    def close(self):
        self.stop_recording()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


class Microphone:
    """Optional microphone: 16-bit WAV recording synced with the log.

    Off unless switched on (--mic, `mic on`, or the panel's checkbox); then
    `log start` also records <session>/audio.wav and `log stop` ends it. The
    device is opened only while recording. Every audio block's arrival time
    (time.time(), the /time/timestamp clock) goes to audio.blocks.csv, so
    sample k of the WAV lines up with the log like a video frame does.
    Never raises on a missing device -- the rest of the app keeps working.
    """

    BLOCK_S = 0.02                  # one callback (and CSV row) per 20 ms

    def __init__(self, device=None, rate=48000, channels=1, enabled=False):
        self.device = device
        self.rate, self.channels = int(rate), int(channels)
        self.enabled = enabled          # record with every `log start`
        self.available = False
        self.message = "microphone not checked"
        self.device_name = None
        self._lock = threading.Lock()
        self._stream = None
        self._queue = None
        self._thread = None
        self._wav = None
        self._blocks_file = None
        self.audio_path = None
        self.record_started = None
        self.samples = self.blocks = self.overflows = 0
        self.level_dbfs = None
        self.error = None

    @staticmethod
    def devices():
        """Input devices as [{index, name, hostapi, rate}], [] without sd."""
        if sd is None:
            return []
        apis = sd.query_hostapis()
        return [{"index": i, "name": d["name"],
                 "hostapi": apis[d["hostapi"]]["name"],
                 "rate": int(d["default_samplerate"])}
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]

    def probe(self, device=None):
        """Look the device up (index, name part, or None = system default)."""
        if device is not None:
            self.device = device
        if sd is None:
            self.message = "sounddevice missing: pip install sounddevice"
            self.available = False
            return False
        try:
            info = sd.query_devices(self.device, "input")
        except Exception as exc:
            self.message = f"no microphone {self.device!r}: {exc}"
            self.available = False
            return False
        self.device_name = info["name"]
        self.available = True
        self.message = f"mic: {self.device_name}"
        return True

    @property
    def recording(self):
        return self._stream is not None

    def _callback(self, data, frames, time_info, status):
        # PortAudio's thread: stamp and hand over, nothing slow here
        self._queue.put((data.copy(), time.time(),
                         bool(status.input_overflow)))

    def _writer_loop(self, wav, rows, queue):
        import numpy as np
        while True:
            item = queue.get()
            if item is None:
                return
            data, stamp, overflow = item
            n = len(data)
            try:
                wav.writeframes(data.tobytes())
                rows.writerow([self.blocks, self.samples, n, f"{stamp:.6f}",
                               f"{stamp - self.record_started:.6f}",
                               int(overflow)])
            except Exception as exc:        # disk full etc.: keep the app up
                self.error = str(exc)
                continue
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) \
                if n else 0.0
            with self._lock:
                self.blocks += 1
                self.samples += n
                self.overflows += int(overflow)
                self.level_dbfs = round(
                    20 * math.log10(max(rms, 1.0) / 32768), 1)

    def start_recording(self, path):
        if not self.available and not self.probe():
            raise RuntimeError(self.message)
        if self.recording:
            raise RuntimeError(f"already recording to {self.audio_path}")
        path = Path(path).with_suffix(".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        queue = Queue()
        try:
            stream = sd.InputStream(
                device=self.device, samplerate=self.rate,
                channels=self.channels, dtype="int16",
                blocksize=int(self.rate * self.BLOCK_S),
                callback=self._callback)
        except Exception as exc:
            self.message = f"cannot open microphone: {exc}"
            raise RuntimeError(self.message) from None
        wav = wave.open(str(path), "wb")
        wav.setnchannels(self.channels)
        wav.setsampwidth(2)
        wav.setframerate(self.rate)
        blocks_file = open(path.with_suffix(".blocks.csv"), "w", newline="",
                           encoding="utf-8")
        rows = csv.writer(blocks_file)
        rows.writerow(["block", "first_sample", "n_samples", "timestamp",
                       "elapsed_s", "overflow"])
        with self._lock:
            self.samples = self.blocks = self.overflows = 0
            self.level_dbfs, self.error = None, None
        self._queue, self._wav, self._blocks_file = queue, wav, blocks_file
        self.audio_path = path
        self._thread = threading.Thread(target=self._writer_loop,
                                        args=(wav, rows, queue),
                                        name="mic-writer", daemon=True)
        self.record_started = time.time()
        self._thread.start()
        try:
            stream.start()
        except Exception as exc:
            stream.close()
            self._stream = stream = None
            self._finish()
            self.message = f"cannot start microphone: {exc}"
            raise RuntimeError(self.message) from None
        self._stream = stream
        self._write_json()
        return str(path)

    def _write_json(self, extra=None):
        info = {
            "audio": self.audio_path.name,
            "blocks_csv": self.audio_path.with_suffix(".blocks.csv").name,
            "device": self.device_name, "sample_rate_hz": self.rate,
            "channels": self.channels, "sample_format": "int16 PCM",
            "start_timestamp": self.record_started,
            "clock": "time.time() (same as HDF5 /time/timestamp)",
            "timing": "blocks.csv timestamp = arrival time of the block on "
                      "the PC, i.e. just after its last sample; sample k is "
                      "at about timestamp(block) - (first_sample + n_samples "
                      "- k) / sample_rate_hz",
        }
        info.update(extra or {})
        self.audio_path.with_suffix(".json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")

    def _finish(self):
        """Drain the writer, close the files; returns the summary."""
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._wav.close()
        self._blocks_file.close()
        self._queue = self._thread = self._wav = self._blocks_file = None
        with self._lock:
            return {"audio_path": str(self.audio_path),
                    "audio_seconds": round(self.samples / self.rate, 3),
                    "audio_blocks": self.blocks,
                    "audio_overflows": self.overflows}

    def stop_recording(self):
        stream = self._stream
        if stream is None:
            return None
        self._stream = None
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        summary = self._finish()
        self._write_json({k: v for k, v in summary.items()
                          if k != "audio_path"})
        if self.error:
            summary["audio_error"] = self.error
        return summary

    def state(self):
        with self._lock:
            return {"available": self.available, "message": self.message,
                    "enabled": self.enabled, "recording": self.recording,
                    "device": self.device, "device_name": self.device_name,
                    "rate_hz": self.rate, "channels": self.channels,
                    "audio_path": (str(self.audio_path) if self.recording
                                   else None),
                    "seconds": round(self.samples / self.rate, 1),
                    "overflows": self.overflows,
                    "level_dbfs": (self.level_dbfs if self.recording
                                   else None),
                    "error": self.error}

    def close(self):
        self.stop_recording()


TICKS_PERIOD_US = 1 << 30      # MicroPython's ticks_us() wraps here (~17.9 min)
LOADCELL_CAL_PATH = Path(__file__).with_name("loadcell_calibration.json")
DEMO_LC_OFFSET, DEMO_LC_SCALE = 84000, 210.0    # simulated counts, counts/g


class LoadCell:
    """HX711 load cell read through the linear guide's Pico.

    The Pico buffers each sample with its own microsecond clock. This side
    drains the buffer whenever the serial link is free (a blocking guide move
    holds it), maps Pico time onto time.time() -- the HDF5 clock -- from the
    request/reply midpoint with the shortest recent round trip, keeps a short
    history for the web graph, and appends to loadcell/ in the open log.
    Never raises on missing hardware: ``available`` is just False.
    """

    HISTORY_S = 15.0
    POLL_S = 0.1
    MAX_DRAINS_PER_POLL = 50        # caps a runaway backlog (e.g. DT shorted)

    def __init__(self, guide=None, demo=False):
        self.guide = guide
        self.demo = demo
        self.source = None              # "pico" | "demo" | None
        self.available = False
        self.message = "load cell not started"
        self.offset = 0.0               # tare, raw counts
        self.scale = None               # raw counts per gram
        self.cal_version = 0            # bumps on tare/calibrate: graph resets
        self.last_log = None            # summary of the last closed log
        self._lock = threading.Lock()           # history + log attachment
        self._poll_lock = threading.Lock()      # one drain sequence at a time
        self._history = deque()         # (seq, host_time, raw, grams)
        self._guide_history = deque()   # (host_time, steps, homed)
        self.steps_per_mm = None
        self._seq = 0
        self._log = None                # (ParamLogger, t0) while recording
        self._dropped = 0
        self._guide_dropped = 0
        self._link_busy = False
        self._last_sample_at = None
        self._last_drain_at = None
        self._board_us = None           # unwrapped Pico clock
        self._board_raw = None
        self._syncs = deque(maxlen=400)     # (host_time, rtt, offset_s)
        self._offset_s = None
        self._demo_t = None
        self._stop = threading.Event()
        self._thread = None

    # ---- setup -----------------------------------------------------------
    def open(self):
        if hasattr(self.guide, "_exec"):            # SerialLinearGuide
            try:
                with self.guide._lock:
                    # An earlier session may have imported the old module and
                    # MicroPython keeps it until reset: reload from flash once.
                    out = self.guide._exec(
                        "import sys\n"
                        "if not hasattr(_guide_module, 'loadcell'):\n"
                        "    sys.modules.pop('snapmarker_linear_module', None)\n"
                        "    import snapmarker_linear_module as _guide_module\n"
                        "    _web_guide = _guide_module.Stepper()\n"
                        "print('__LCMOD__|%d|%s' % (hasattr(_guide_module, "
                        "'loadcell'), getattr(_guide_module, '__file__', '?')))")
                    marker = next(line for line in out.splitlines()
                                  if line.startswith("__LCMOD__|"))
                    _, has_loadcell, board_file = marker.split("|", 2)
                    if has_loadcell != "1":
                        raise RuntimeError(
                            f"{board_file} on the Pico is the old version")
                    out = self.guide._exec(
                        "_web_lc = _guide_module.loadcell(_web_guide)\n"
                        "print('__SPM__|%r' % _guide_module.STEPS_PER_MM)")
                self.steps_per_mm = float(next(
                    line for line in out.splitlines()
                    if line.startswith("__SPM__|")).split("|")[1])
            except Exception as exc:
                self.message = ("Pico has no load cell support -- copy the "
                                "updated snapmarker_linear_module.py to it "
                                f"({exc})")
                return False
            self.source = "pico"
            self.message = "HX711 on guide Pico (waiting for samples)"
            self._load_calibration()
        elif self.demo:
            self.source = "demo"
            self.offset, self.scale = float(DEMO_LC_OFFSET), DEMO_LC_SCALE
            self.steps_per_mm = 400.0       # 200 steps x 16 microsteps / 8 mm
            self.message = "simulated load cell (--demo)"
        else:
            self.message = ("load cell needs the guide Pico: start with "
                            "--guide-port COMx")
            return False
        self.available = True
        self._thread = threading.Thread(target=self._loop, name="loadcell",
                                        daemon=True)
        self._thread.start()
        return True

    def _load_calibration(self):
        try:
            data = json.loads(LOADCELL_CAL_PATH.read_text(encoding="utf-8"))
            self.offset = float(data["offset_counts"])
            self.scale = float(data["counts_per_g"]) or None
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save_calibration(self):
        if self.source != "pico":       # demo values must not replace real ones
            return
        LOADCELL_CAL_PATH.write_text(json.dumps({
            "offset_counts": self.offset, "counts_per_g": self.scale,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
            encoding="utf-8")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.source == "pico" and self.guide._lock.acquire(timeout=2.0):
            try:        # never leave core 1 running when the PC lets go
                self.guide._exec("_web_lc.stop()")
            except Exception:
                pass
            finally:
                self.guide._lock.release()

    # ---- sampling --------------------------------------------------------
    def _loop(self):
        while not self._stop.wait(self.POLL_S):
            self.poll()

    def poll(self, wait=False):
        """Fetch everything buffered so far. ``wait`` waits for the link."""
        if not self.available:
            return 0
        total = 0
        with self._poll_lock:
            for _ in range(self.MAX_DRAINS_PER_POLL):
                try:
                    samples, positions, backlog = (
                        self._fetch_pico(wait) if self.source == "pico"
                        else self._fetch_demo())
                except Exception as exc:
                    self.message = f"load cell read error: {exc}"
                    return total
                if samples or positions:
                    self._ingest(samples, positions)
                    total += len(samples)
                if not backlog:
                    break
        self._check_health()
        return total

    def _fetch_pico(self, wait):
        lock = self.guide._lock
        got = lock.acquire(timeout=2.0) if wait else lock.acquire(blocking=False)
        if not got:
            self._link_busy = True      # guide move: the Pico keeps buffering
            return [], [], 0
        try:
            t_send = time.time()
            out = self.guide._exec("_web_lc.drain()")
            t_recv = time.time()
        finally:
            lock.release()
        self._link_busy = False
        self._last_drain_at = t_recv
        lines = out.splitlines()
        head = next(i for i, line in enumerate(lines)
                    if line.startswith("__LC__|"))
        now_raw, dropped, backlog, guide_dropped, guide_backlog = (
            int(x) for x in lines[head].split("|")[1:6])
        if self._board_raw is None:
            self._board_us = now_raw
        else:
            self._board_us += (now_raw - self._board_raw) % TICKS_PERIOD_US
        self._board_raw = now_raw
        self._sync(t_send, t_recv)
        self._dropped, self._guide_dropped = dropped, guide_dropped

        def rows(line):
            """``ticks,v[,v]`` items -> (host_time, int, ...) tuples."""
            out = []
            for item in line.split(";"):
                if item:
                    stamp, *values = item.split(",")
                    age_us = (now_raw - int(stamp)) % TICKS_PERIOD_US
                    out.append((self._offset_s
                                + (self._board_us - age_us) / 1e6,
                                *(int(v) for v in values)))
            return out

        body = lines[head + 1:] + ["", ""]
        return rows(body[0]), rows(body[1]), backlog or guide_backlog

    def _sync(self, t_send, t_recv):
        """Pico->PC clock offset from the cleanest recent exchange."""
        self._syncs.append((t_recv, t_recv - t_send,
                            (t_send + t_recv) / 2 - self._board_us / 1e6))
        recent = [s for s in self._syncs if t_recv - s[0] < 30.0]
        self._offset_s = min(recent, key=lambda s: s[1])[2]

    def _fetch_demo(self):
        now = time.time()
        if self._demo_t is None:
            self._demo_t = self._demo_guide_t = now
        samples = []
        while self._demo_t + 1 / 80 <= now:
            self._demo_t += 1 / 80
            grams = (300 * max(0.0, math.sin(self._demo_t * 0.7))
                     + random.gauss(0, 3))
            samples.append((self._demo_t,
                            int(DEMO_LC_OFFSET + grams * DEMO_LC_SCALE)))
        positions = []
        mm = getattr(self.guide, "position_mm", None)
        while mm is not None and self._demo_guide_t + 0.02 <= now:
            self._demo_guide_t += 0.02
            positions.append((self._demo_guide_t,
                              int(round(mm * self.steps_per_mm)),
                              int(bool(self.guide.homed))))
        return samples, positions, 0

    def to_grams(self, raw):
        return (raw - self.offset) / self.scale if self.scale else None

    def _ingest(self, samples, positions=()):
        rows = []
        with self._lock:
            for stamp, raw in samples:
                grams = self.to_grams(raw)
                self._seq += 1
                self._history.append((self._seq, stamp, raw, grams))
                rows.append((stamp, raw, grams))
            self._guide_history.extend(positions)
            for history, key in ((self._history, 1), (self._guide_history, 0)):
                if history:
                    cutoff = history[-1][key] - self.HISTORY_S
                    while history[0][key] < cutoff:
                        history.popleft()
            if samples:
                self._last_sample_at = time.time()
            log = self._log
        if log is not None:
            self._write(log, rows, positions)

    def _rate(self):
        """Samples per second over the last two seconds of history."""
        if len(self._history) < 3:
            return None
        end = self._history[-1][1]
        times = [h[1] for h in self._history if h[1] >= end - 2.0]
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else None

    def _check_health(self):
        if self.source != "pico":
            return
        now = time.time()
        stale = self._last_sample_at is None or now - self._last_sample_at > 2
        if stale and self._last_drain_at and now - self._last_drain_at < 1:
            self.message = ("no HX711 samples: check DT=GP17, SCK=GP16, "
                            "VCC and GND")
        elif not stale:
            with self._lock:
                rate = self._rate()
            if rate and rate > 100:
                self.message = (f"HX711 data rate {rate:.0f} Hz is not "
                                "possible: check the DT wire")
            else:
                self.message = "HX711 on guide Pico"

    # ---- HDF5 ------------------------------------------------------------
    @property
    def recording(self):
        return self._log is not None

    def _cal_attrs(self, group):
        group.attrs["offset_counts"] = float(self.offset)
        group.attrs["counts_per_g"] = (float(self.scale) if self.scale
                                       else float("nan"))
        group.attrs["force_g_formula"] = "(raw - offset_counts) / counts_per_g"

    def attach_log(self, logger):
        """Start appending to loadcell/ in the log ``logger`` just opened."""
        if not self.available or logger._file is None:
            return
        with logger._mark_lock:
            file = logger._file
            group = file.require_group("loadcell")
            for name, dtype, units in (
                    ("timestamp", "f8", "seconds since 1970-01-01 00:00:00 UTC"),
                    ("elapsed_s", "f8", "seconds since log start"),
                    ("raw", "i4", "HX711 counts"),
                    ("force_g", "f4", "grams (NaN until calibrated)")):
                dataset = group.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype,
                    chunks=(1024,), compression="gzip", compression_opts=1)
                dataset.attrs["units"] = units
            group.attrs["sensor"] = ("HX711 channel A gain 128 on the linear "
                                     "guide Pico (DT=GP17, SCK=GP16)")
            group.attrs["source"] = self.source
            group.attrs["clock"] = ("Pico ticks_us mapped to time.time() "
                                    "(same as /time/timestamp)")
            self._cal_attrs(group)
            guide = file.require_group("guide")
            for name, dtype, units in (
                    ("timestamp", "f8", "seconds since 1970-01-01 00:00:00 UTC"),
                    ("elapsed_s", "f8", "seconds since log start"),
                    ("position_mm", "f4", "mm from the home switch"),
                    ("position_steps", "i4", "microsteps from the home switch"),
                    ("homed", "u1", "1 = position is referenced to home")):
                dataset = guide.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype,
                    chunks=(1024,), compression="gzip", compression_opts=1)
                dataset.attrs["units"] = units
            guide.attrs["steps_per_mm"] = float(self.steps_per_mm or "nan")
            guide.attrs["rate_hz"] = 50.0
            guide.attrs["source"] = self.source
            guide.attrs["note"] = ("open-loop commanded step count recorded "
                                   "on the Pico during moves; only meaningful "
                                   "where homed == 1")
            marks = file["marks"].require_group("loadcell")
            for name, dtype in (("raw", "i4"), ("force_g", "f4"),
                                ("sample_timestamp", "f8")):
                marks.create_dataset(name, shape=(0,), maxshape=(None,),
                                     dtype=dtype, chunks=True)
            marks = file["marks"].require_group("guide")
            for name, dtype in (("position_mm", "f4"), ("homed", "u1"),
                                ("sample_timestamp", "f8")):
                marks.create_dataset(name, shape=(0,), maxshape=(None,),
                                     dtype=dtype, chunks=True)
        with self._lock:
            t0 = logger._t0
            # samples already fetched since the log began
            backlog = [(h[1], h[2], h[3]) for h in self._history if h[1] >= t0]
            positions = [p for p in self._guide_history if p[0] >= t0]
            self._log = (logger, t0)
        self._write(self._log, backlog, positions)

    @staticmethod
    def _extend(group, columns):
        """Append equal-length lists, {dataset name: values}, to ``group``."""
        count = len(next(iter(columns.values())))
        start = group["timestamp"].shape[0]
        for name, values in columns.items():
            group[name].resize((start + count,))
            group[name][start:] = values

    def _write(self, log, rows, positions=()):
        logger, t0 = log
        rows = [row for row in rows if row[0] >= t0]
        positions = [p for p in positions if p[0] >= t0]
        if not rows and not positions:
            return
        try:
            with logger._mark_lock:
                file = logger._file
                if file is None:
                    return
                if rows:
                    self._extend(file["loadcell"], {
                        "timestamp": [r[0] for r in rows],
                        "elapsed_s": [r[0] - t0 for r in rows],
                        "raw": [r[1] for r in rows],
                        "force_g": [r[2] if r[2] is not None
                                    else float("nan") for r in rows]})
                if positions:
                    spm = self.steps_per_mm or float("nan")
                    self._extend(file["guide"], {
                        "timestamp": [p[0] for p in positions],
                        "elapsed_s": [p[0] - t0 for p in positions],
                        "position_mm": [p[1] / spm for p in positions],
                        "position_steps": [p[1] for p in positions],
                        "homed": [p[2] for p in positions]})
        except Exception as exc:        # never take the hand logger down
            print(f"  [loadcell] HDF5 write error: {exc}")

    def detach_log(self):
        """Pull what the Pico still holds, then stop writing to the log."""
        if self._log is None:
            return None
        self.poll(wait=True)
        with self._lock:
            logger, _ = self._log
            self._log = None
        with logger._mark_lock:
            file = logger._file
            count = (int(file["loadcell/timestamp"].shape[0])
                     if file is not None else 0)
            positions = (int(file["guide/timestamp"].shape[0])
                         if file is not None else 0)
            if file is not None:
                file["loadcell"].attrs["n_samples"] = count
                file["loadcell"].attrs["dropped_on_pico"] = int(self._dropped)
                file["guide"].attrs["n_samples"] = positions
                file["guide"].attrs["dropped_on_pico"] = int(
                    self._guide_dropped)
        self.last_log = {"loadcell_samples": count,
                         "guide_position_samples": positions}
        return self.last_log

    def add_mark(self, logger, result):
        """Store the latest load-cell sample and stage position with a mark.

        Both groups always get a row (NaN when there is no data yet), so row
        k of marks/loadcell and marks/guide stays mark k.
        """
        with self._lock:
            latest = self._history[-1] if self._history else None
            position = self._guide_history[-1] if self._guide_history else None
            attached = self._log is not None
        if not attached:
            return
        nan = float("nan")
        _, stamp, raw, grams = latest or (None, nan, 0, None)
        p_stamp, steps, homed = position or (nan, None, 0)
        mm = (steps / self.steps_per_mm
              if steps is not None and self.steps_per_mm else None)
        with logger._mark_lock:
            for path, values in (
                    ("marks/loadcell", {"raw": raw, "sample_timestamp": stamp,
                                        "force_g": nan if grams is None
                                        else grams}),
                    ("marks/guide", {"position_mm": nan if mm is None else mm,
                                     "homed": homed,
                                     "sample_timestamp": p_stamp})):
                group = logger._file[path]
                for name, value in values.items():
                    index = group[name].shape[0]
                    group[name].resize((index + 1,))
                    group[name][index] = value
        result["loadcell_g"] = round(grams, 2) if grams is not None else None
        result["loadcell_raw"] = raw if latest else None
        result["guide_mm"] = round(mm, 3) if mm is not None else None

    # ---- calibration -----------------------------------------------------
    def _mean_raw(self, seconds=1.0):
        self.poll(wait=True)
        with self._lock:
            if not self._history:
                raise RuntimeError(f"no load cell data: {self.message}")
            end = self._history[-1][1]
            values = [h[2] for h in self._history if h[1] >= end - seconds]
        if time.time() - end > 2.0 or len(values) < 3:
            raise RuntimeError("no recent load cell samples to average")
        return sum(values) / len(values), len(values)

    def _calibration_changed(self):
        self.cal_version += 1
        self._save_calibration()
        with self._lock:
            log = self._log
        if log is not None:
            with log[0]._mark_lock:
                self._cal_attrs(log[0]._file["loadcell"])

    def tare(self):
        self.offset, count = self._mean_raw()
        self._calibration_changed()
        return {"offset_counts": round(self.offset, 1), "averaged": count}

    def calibrate(self, grams):
        if grams <= 0:
            raise ValueError("calibration weight must be more than 0 g")
        raw, count = self._mean_raw()
        counts = raw - self.offset
        if abs(counts) < 50:
            raise RuntimeError("reading barely changed since tare: tare with "
                               "nothing on the cell, then add the weight")
        self.scale = counts / grams
        self._calibration_changed()
        return {"counts_per_g": round(self.scale, 3),
                "reading_g": round(grams, 1), "averaged": count}

    # ---- web -------------------------------------------------------------
    def state(self):
        with self._lock:
            latest = self._history[-1] if self._history else None
            position = self._guide_history[-1] if self._guide_history else None
            rate = self._rate()
        grams = self.to_grams(latest[2]) if latest else None
        return {
            "available": self.available, "message": self.message,
            "source": self.source, "recording": self.recording,
            "link_busy": self._link_busy,
            "raw": latest[2] if latest else None,
            "grams": round(grams, 2) if grams is not None else None,
            "calibrated": self.scale is not None,
            "offset_counts": round(self.offset, 1),
            "counts_per_g": round(self.scale, 4) if self.scale else None,
            "rate_hz": round(rate, 1) if rate else None,
            "rate_pin": (None if not rate else "high (80 Hz)" if rate > 40
                         else "low (10 Hz)"),
            "dropped": self._dropped,
            "guide_mm": (round(position[1] / self.steps_per_mm, 3)
                         if position and self.steps_per_mm else None),
        }

    def web_payload(self, since=0):
        """Samples newer than ``since`` in current units, for the graph."""
        with self._lock:
            rows = [h for h in self._history if h[0] > since]
        calibrated = self.scale is not None
        return {
            "state": self.state(), "cal_version": self.cal_version,
            "units": "g" if calibrated else "raw",
            "samples": [[seq, round(stamp, 4),
                         round(self.to_grams(raw), 2) if calibrated else raw]
                        for seq, stamp, raw, _ in rows],
        }


class AESensor:
    """Two acoustic-emission envelope channels on Board 2, streamed over USB.

    Board 2 is found by USB serial (rig_boards.json) and its file list is
    checked before anything is sent (see ae_board2.py). Mode "ram" pastes
    ae_ram_sampler.py into RAM (64-sample frames, CRC-16); mode "legacy"
    reads the untouched main.py stream. Nothing is ever written to the
    board, and disconnecting soft-resets it back to main.py.

    Sampling runs whenever connected, for the live view. While a log is open
    every sample goes into ae/ of the same .h5: raw counts in ae/data (row k
    is sample first_sample_index + k; gaps hold 65535) and, per frame, the
    first sample index and the time.time() it arrived -- the clock of
    /time/timestamp. At log stop a line t_pc = clock_a * index + clock_b is
    fit to the earliest arrivals and stored as attributes. The browser only
    gets 20 ms RMS/max summaries. Never raises on missing hardware.
    """

    HISTORY_S = 10.0
    RMS_WINDOW = 400                # samples per live point: 20 ms, 50/s
    WRITE_S = 1.0                   # buffered seconds per HDF5 write
    RECENT_S = 3.0                  # frames kept for a log that just began
    MAX_GAP_S = 600.0               # a bigger index jump = stream restarted

    def __init__(self, mode="ram", port=None):
        self.mode_wanted = mode
        self.port_override = port or None
        self.link = None
        self.available = False
        self.message = "AE not started"
        self.info = {}
        self.channels = {}
        self.last_log = None
        self._logger = None
        self._lock = threading.Lock()       # history, counters, log buffer
        self._write_lock = threading.Lock()
        self._history = deque()     # (seq, t, rms0, rms1, max0, max1) in mV
        self._recent = deque()      # recent frames for attach_log
        self._acc = []
        self._acc_n = 0
        self._seq = 0
        self._log = None
        self._stop = threading.Event()
        self._thread = self._writer = None
        self._reset_counters()

    def _reset_counters(self):
        self.frames = self.dropped = self.overruns = self.fifo_flags = 0
        self.index_gaps = 0
        self._last_seq = self._next_index = None

    # ---- connection ------------------------------------------------------
    def open(self, mode=None, logger=None):
        self.close()
        mode = mode or self.mode_wanted
        try:
            import ae_board2
            config = ae_board2.load_config()
            self.channels = config.get("ae_channels", {})
            self.message = "connecting to Board 2…"
            link = ae_board2.open_link(mode, self.port_override, config)
            link.start()
        except Exception as exc:
            self.message = f"Board 2 not used: {exc}"
            return False
        self.link, self.info = link, dict(link.info)
        self.mode_wanted = mode
        self._reset_counters()
        with self._lock:
            self._history.clear()
            self._recent.clear()
            self._acc, self._acc_n = [], 0
        self.available = True
        self.message = (f"Board 2 on {link.port} · {link.mode} · "
                        f"{self.info.get('fs_hz')} Hz × 2 ch")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ae-reader",
                                        daemon=True)
        self._thread.start()
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(target=self._writer_loop,
                                            name="ae-writer", daemon=True)
            self._writer.start()
        logger = logger or self._logger
        if logger is not None and logger.active and self._log is None:
            self.attach_log(logger)         # a log already running
        return True

    def close(self):
        """Stop streaming and hand Board 2 back to its untouched main.py."""
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        link, self.link = self.link, None
        if link is not None:
            link.close()
            self.message = ("disconnected · Board 2 is back on its old AE "
                            "firmware (main.py)")
        self.available = False

    def _loop(self):
        link = self.link
        while not self._stop.is_set():
            try:
                frames = link.read()
            except Exception as exc:        # unplugged / port gone
                self.available = False
                self.message = (f"Board 2 lost ({exc}); a reset returns it "
                                "to its old AE firmware (main.py)")
                return
            if frames:
                self._ingest(frames)

    def _ingest(self, frames):
        fs = self.info.get("fs_hz") or 20000
        with self._lock:
            for f in frames:
                self.frames += 1
                if (self._last_seq is not None
                        and f.seq != (self._last_seq + 1) & 0xFFFFFFFF):
                    self.dropped += (f.seq - self._last_seq - 1) & 0xFFFFFFFF
                self._last_seq = f.seq
                if self._next_index is not None and f.first != self._next_index:
                    self.index_gaps += 1
                self._next_index = f.first + f.n
                if f.flags & 1 and not f.flags & 4:
                    self.overruns += 1
                if f.flags & 2:
                    self.fifo_flags += 1
                self._recent.append(f)
                if self._log is not None and f.stamp >= self._log["t0"]:
                    self._log["pending"].append(f)
                    self._log["pending_n"] += f.n
                self._acc.append(f.samples)
                self._acc_n += f.n
                if self._acc_n >= self.RMS_WINDOW:
                    self._summarise(f.stamp, fs)
            cutoff = frames[-1].stamp - self.RECENT_S
            while self._recent and self._recent[0].stamp < cutoff:
                self._recent.popleft()
            cutoff = frames[-1].stamp - self.HISTORY_S
            while self._history and self._history[0][1] < cutoff:
                self._history.popleft()

    def _summarise(self, stamp, fs):
        """Turn whole RMS windows of buffered samples into live points."""
        import numpy as np
        data = np.concatenate(self._acc).astype(np.float64)
        whole = len(data) // self.RMS_WINDOW * self.RMS_WINDOW
        mv = data[:whole].reshape(-1, self.RMS_WINDOW, 2) * (3300.0 / 4095)
        rms = np.sqrt((mv ** 2).mean(axis=1))
        peak = mv.max(axis=1)
        count = len(rms)
        for i in range(count):
            self._seq += 1
            t = stamp - (count - 1 - i) * self.RMS_WINDOW / fs
            self._history.append((self._seq, t, *(round(float(v), 2) for v in
                                  (rms[i, 0], rms[i, 1], peak[i, 0],
                                   peak[i, 1]))))
        rest = data[whole:].astype(np.uint16)
        self._acc, self._acc_n = ([rest] if len(rest) else []), len(rest)

    # ---- HDF5 ------------------------------------------------------------
    @property
    def recording(self):
        return self._log is not None

    def _static_attrs(self, group):
        import ae_board2 as ab
        info, link = self.info, self.link
        attrs = group.attrs
        attrs["mode"] = link.mode
        attrs["code_version"] = str(info.get("version", "?"))
        attrs["board_info"] = json.dumps(info)
        attrs["board_usb_serial"] = link.serial_no
        attrs["board_port"] = link.port
        attrs["fs_hz"] = float(info.get("fs_hz", ab.FS_HZ))
        attrs["adc_bits"] = int(info.get("adc_bits", ab.ADC_BITS))
        attrs["pins"] = json.dumps(["GP26", "GP27"])
        attrs["channel_names"] = json.dumps(["ch0", "ch1"])
        for key in ("ch0", "ch1"):
            ch = self.channels.get(key, {})
            attrs[f"{key}_pin"] = ch.get("pin", "GP26" if key == "ch0"
                                         else "GP27")
            attrs[f"{key}_old_label"] = ch.get("old_label", "?")
            attrs[f"{key}_mounting"] = ch.get("mounting", "UNSPECIFIED")
        attrs["channel_skew_s"] = 25e-6
        attrs["channel_skew_note"] = ("round-robin ADC: in each pair ch1 is "
                                      "converted 25 us after ch0")
        attrs["units"] = ("raw ADC counts, 12-bit (0-4095); 65535 = no "
                          "sample (gap)")
        attrs["missing_value"] = ab.MISSING
        attrs["sensor_model"] = "AE-ENV24A10"
        attrs["signal"] = ("0-2 kHz ENVELOPE of the acoustic emission "
                           "(module output 0-10 V), not the raw AE waveform")
        attrs["volts_per_count_at_pin"] = ab.VREF_V / ab.FULL_SCALE_CODE
        attrs["volts_per_count_at_pin_source"] = (
            "VREF = 3.3 from Board 2 config.py; / 4095 (12-bit max code) "
            "following the old firmware's raw * VREF / max_code convention "
            "(preamp_mod.py)")
        attrs["module_volts_per_pin_volt"] = ab.DIVIDER_GAIN
        attrs["module_volts_per_pin_volt_source"] = (
            "ae_config.py: module output 0-10 Vdc scaled to 0-3.2 V by the "
            "onboard 100k/47k divider -> (100k + 47k) / 47k")
        attrs["volts_per_count_module_output"] = (ab.VREF_V / ab.FULL_SCALE_CODE
                                                  * ab.DIVIDER_GAIN)
        attrs["conversion_formula"] = (
            "V_pin = counts * volts_per_count_at_pin; V_module = V_pin * "
            "module_volts_per_pin_volt")
        attrs["conversion_note"] = (
            "calib_config.json of the old app holds only timing, no voltage "
            "factor. The old GUI's exported volts equal counts * 3.3 / 65535 "
            "applied to these 12-bit counts, i.e. 16x too small.")
        attrs["clock"] = ("frame_pc_time = time.time() right after the "
                          "serial read returned (same clock as "
                          "/time/timestamp)")
        attrs["sample_index_note"] = ("sample index of ae/data row k = "
                                      "first_sample_index + k")

    def attach_log(self, logger):
        """Start recording into ae/ of the log ``logger`` just opened."""
        self._logger = logger
        if not self.available or logger._file is None:
            return
        with logger._mark_lock:
            file = logger._file
            if "ae" in file:        # reconnect mid-log: the index restarted
                self.message += " · reconnected: records from the next log"
                return
            group = file.create_group("ae")
            data = group.create_dataset(
                "data", shape=(0, 2), maxshape=(None, 2), dtype="u2",
                chunks=(16384, 2), compression="gzip", compression_opts=1,
                shuffle=True)
            data.attrs["units"] = "raw ADC counts (12-bit); 65535 = gap"
            data.attrs["columns"] = json.dumps(["ch0 GP26", "ch1 GP27"])
            for name, dtype, units in (
                    ("frame_first_index", "u8", "sample index of the frame's "
                     "first sample (since the board's START)"),
                    ("frame_pc_time", "f8", "time.time() at arrival, seconds "
                     "since 1970-01-01 00:00:00 UTC"),
                    ("frame_seq", "u4", "board frame counter"),
                    ("frame_n_samples", "u2", "samples per channel in frame"),
                    ("frame_flags", "u2", "bit0 overrun gap before frame, "
                     "bit1 ADC FIFO overflow, bit2 first frame after START")):
                dataset = group.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype,
                    chunks=(4096,), compression="gzip", compression_opts=1)
                dataset.attrs["units"] = units
            self._static_attrs(group)
        self.last_log = None
        with self._lock:
            t0 = logger._t0
            pending = [f for f in self._recent if f.stamp >= t0]
            self._log = {"logger": logger, "t0": t0, "pending": pending,
                         "pending_n": sum(f.n for f in pending),
                         "next_index": None, "first_index": None,
                         "rows": 0, "frames": 0, "gap_samples": 0,
                         "restarted": False, "dropped0": self.dropped,
                         "overruns0": self.overruns,
                         "fifo0": self.fifo_flags,
                         "bad0": self._bad_frames()}

    def _bad_frames(self):
        link = self.link
        return link.parser.bad_frames if link and link.parser else 0

    def _writer_loop(self):
        while True:
            time.sleep(0.5)
            log = self._log
            if log is not None and log["pending_n"] >= (
                    (self.info.get("fs_hz") or 20000) * self.WRITE_S):
                self._write_pending(log)

    def _write_pending(self, log):
        with self._write_lock:          # writer thread vs detach_log
            self._write_pending_locked(log)

    def _write_pending_locked(self, log):
        import numpy as np
        import ae_board2 as ab
        with self._lock:
            frames, log["pending"], log["pending_n"] = log["pending"], [], 0
        if not frames or log["restarted"]:
            return
        fs = self.info.get("fs_hz") or 20000
        blocks, kept = [], []
        index = log["next_index"]
        for f in frames:
            if index is None:
                index = log["first_index"] = f.first
            if f.first < index or f.first - index > fs * self.MAX_GAP_S:
                log["restarted"] = True     # board restarted mid-log
                break
            if f.first > index:
                gap = f.first - index
                blocks.append(np.full((gap, 2), ab.MISSING, np.uint16))
                log["gap_samples"] += gap
            blocks.append(f.samples)
            kept.append(f)
            index = f.first + f.n
        log["next_index"] = index
        if not kept:
            return
        data = np.concatenate(blocks)
        columns = {
            "frame_first_index": [f.first for f in kept],
            "frame_pc_time": [f.stamp for f in kept],
            "frame_seq": [f.seq for f in kept],
            "frame_n_samples": [f.n for f in kept],
            "frame_flags": [f.flags for f in kept]}
        logger = log["logger"]
        try:
            with logger._mark_lock:
                file = logger._file
                if file is None:
                    return
                group = file["ae"]
                start = group["data"].shape[0]
                group["data"].resize((start + len(data), 2))
                group["data"][start:] = data
                count = group["frame_pc_time"].shape[0]
                for name, values in columns.items():
                    group[name].resize((count + len(kept),))
                    group[name][count:] = values
                if log["first_index"] is not None:
                    group.attrs["first_sample_index"] = log["first_index"]
            log["rows"] += len(data)
            log["frames"] += len(kept)
        except Exception as exc:        # never take the hand logger down
            print(f"  [ae] HDF5 write error: {exc}")

    def detach_log(self):
        """Write what is buffered, fit the clock, stop writing to the log."""
        import numpy as np
        import ae_board2 as ab
        log = self._log
        if log is None:
            return None
        time.sleep(0.05)                # let in-flight frames land
        self._write_pending(log)
        with self._lock:
            self._log = None
        summary = {"ae_samples": log["rows"], "ae_frames": log["frames"]}
        logger = log["logger"]
        with logger._mark_lock:
            file = logger._file
            if file is None or "ae" not in file:
                return summary
            group = file["ae"]
            counts = {
                "n_samples": log["rows"], "n_frames": log["frames"],
                "dropped_frames": self.dropped - log["dropped0"],
                "bad_frames": self._bad_frames() - log["bad0"],
                "overruns": self.overruns - log["overruns0"],
                "fifo_overflow_frames": self.fifo_flags - log["fifo0"],
                "gap_samples": log["gap_samples"],
                "stream_restarted": int(log["restarted"])}
            for key, value in counts.items():
                group.attrs[key] = int(value)
            fit = None
            if log["frames"] >= 2:
                ends = (group["frame_first_index"][:].astype(np.float64)
                        + group["frame_n_samples"][:])
                fit = ab.fit_clock(ends, group["frame_pc_time"][:],
                                   float(group.attrs["fs_hz"]))
            if fit:
                for key, value in fit.items():
                    group.attrs[key if key.startswith("clock_")
                                else f"clock_fit_{key}"] = value
                group.attrs["clock_formula"] = (
                    "t_pc = clock_a * sample_index + clock_b, sample_index = "
                    "first_sample_index + row; t_pc on the /time/timestamp "
                    "clock")
                group.attrs["clock_offset_note"] = (
                    "fit to the lower envelope of frame arrivals (earliest "
                    "per window), so t_pc is late by the minimum USB + host "
                    "latency: a constant, unknown offset of a few ms at most")
                summary.update({
                    "ae_fs_measured_hz": round(fit["fs_measured_hz"], 3),
                    "ae_clock_residual_ms": round(fit["residual_rms_ms"], 3)})
            else:
                group.attrs["clock_fit"] = "not enough frames"
            summary.update({k: counts[k] for k in
                            ("dropped_frames", "bad_frames", "overruns")})
        self.last_log = summary
        return summary

    # ---- web -------------------------------------------------------------
    def _rate(self):
        recent = list(self._recent)
        if len(recent) < 2 or recent[-1].stamp - recent[0].stamp < 0.5:
            return None
        n = sum(f.n for f in recent[1:])
        return n / (recent[-1].stamp - recent[0].stamp)

    def state(self):
        link = self.link
        with self._lock:
            rate = self._rate() if self.available else None
            latest = self._history[-1] if self._history else None
        return {
            "available": self.available, "message": self.message,
            "mode": link.mode if link else None,
            "port": link.port if link else None,
            "serial": link.serial_no if link else None,
            "recording": self.recording,
            "fs_hz": self.info.get("fs_hz"),
            "rate_hz": round(rate) if rate else None,
            "frames": self.frames, "dropped": self.dropped,
            "bad": self._bad_frames(), "overruns": self.overruns,
            "rms_mv": list(latest[2:4]) if latest else None,
            "channels": {k: f"{v.get('pin')} {v.get('old_label')}"
                         for k, v in self.channels.items()},
        }

    def web_payload(self, since=0):
        """RMS/max per 20 ms window newer than ``since`` -- never raw AE."""
        with self._lock:
            points = [list(h) for h in self._history if h[0] > since]
        return {"state": self.state(), "points": points}


def hook_logger(logger, loadcell, ae=None):
    """Make every log start/stop/mark carry the load cell (and AE), whoever
    calls it.

    ParamLogger.start() calls self.stop() first, which resolves to the
    wrapped stop below, so a restarted log is detached before it is closed.
    """
    start, stop, mark = logger.start, logger.stop, logger.mark

    def start_with_loadcell(path=None):
        result = start(path)
        loadcell.attach_log(logger)
        if ae is not None:
            ae.attach_log(logger)
        return result

    def stop_with_loadcell():
        if ae is not None:
            ae.detach_log()
        loadcell.detach_log()           # needs the file still open
        return stop()

    def mark_with_loadcell(label=None):
        result = mark(label)
        loadcell.add_mark(logger, result)
        return result

    logger.start, logger.stop = start_with_loadcell, stop_with_loadcell
    logger.mark = mark_with_loadcell


class CamCommandRunner(CommandRunner):
    """Adds `cam ...`, `mode ...`, `lc ...` and `ae ...` commands and ties
    video start/stop to `log start/stop`."""

    def __init__(self, fc, mode, guide, camera, loadcell, ae=None, mic=None):
        super().__init__(fc, mode, guide)
        self.camera = camera
        self.loadcell = loadcell
        self.ae = ae
        self.mic = mic if mic is not None else Microphone()

    def _mic_command(self, tokens):
        mic = self.mic
        sub = tokens[1].lower() if len(tokens) > 1 else "status"
        if sub == "status" and len(tokens) <= 2:
            return mic.state()
        if sub == "devices" and len(tokens) == 2:
            return mic.devices()
        if sub in ("on", "off") and len(tokens) == 2:
            mic.enabled = sub == "on"
            if mic.enabled and not mic.available:
                mic.probe()
            return mic.state()
        if sub == "device" and len(tokens) >= 3:
            if mic.recording:
                raise RuntimeError("stop the recording before changing device")
            name = " ".join(tokens[2:])
            device = None if name.lower() == "default" else (
                int(name) if name.isdigit() else name)
            if not mic.probe(device):
                raise RuntimeError(mic.message)
            return mic.state()
        if sub == "start" and len(tokens) <= 3:
            name = tokens[2] if len(tokens) > 2 else time.strftime(
                "audio_%Y%m%d_%H%M%S")
            folder = self.fc.logger.session_dir if self.fc.logger.active \
                else Path("data_logs") / "audio"
            return {"audio_path": mic.start_recording(Path(folder) / name)}
        if sub == "stop" and len(tokens) == 2:
            return mic.stop_recording() or "Not recording."
        raise ValueError("usage: mic status | mic devices | mic on | mic off | "
                         "mic device <index|name|default> | mic start [name] | "
                         "mic stop")

    def _ae_command(self, tokens):
        sub = tokens[1].lower() if len(tokens) > 1 else "status"
        if sub == "status" and len(tokens) <= 2:
            return self.ae.state()
        if sub == "ports" and len(tokens) == 2:
            import ae_board2
            return ae_board2.list_boards()
        if sub == "info" and len(tokens) == 2:
            if self.ae.link is None:
                raise RuntimeError(f"AE not connected: {self.ae.message}")
            return self.ae.info
        if sub == "connect" and len(tokens) <= 3:
            mode = tokens[2].lower() if len(tokens) == 3 else "ram"
            if mode not in ("ram", "legacy"):
                raise ValueError("usage: ae connect [ram|legacy]")
            if not self.ae.open(mode, self.fc.logger):
                raise RuntimeError(self.ae.message)
            return self.ae.state()
        if sub == "disconnect" and len(tokens) == 2:
            self.ae.close()
            return self.ae.message
        raise ValueError("usage: ae status | ae ports | ae info | "
                         "ae connect [ram|legacy] | ae disconnect")

    def _loadcell_command(self, tokens):
        sub = tokens[1].lower() if len(tokens) > 1 else "status"
        if sub == "status" and len(tokens) <= 2:
            return self.loadcell.state()
        if sub == "tare" and len(tokens) == 2:
            return self.loadcell.tare()
        if sub in ("calibrate", "cal") and len(tokens) == 3:
            try:
                grams = float(tokens[2])
            except ValueError:
                raise ValueError("calibration weight must be a number "
                                 "of grams") from None
            return self.loadcell.calibrate(grams)
        raise ValueError("usage: lc status | lc tare | lc calibrate <grams>")

    def _cam_command(self, tokens):
        sub = tokens[1].lower() if len(tokens) > 1 else "status"
        if sub == "status":
            return self.camera.state()
        if sub == "start":
            name = tokens[2] if len(tokens) > 2 else time.strftime(
                "video_%Y%m%d_%H%M%S")
            folder = self.fc.logger.session_dir if self.fc.logger.active \
                else Path("data_logs") / "video"
            return {"video_path": self.camera.start_recording(
                Path(folder) / name)}
        if sub == "stop":
            return self.camera.stop_recording() or "Not recording."
        raise ValueError("usage: cam start [name] | cam stop | cam status")

    def _mode_command(self, tokens):
        fc = self.fc
        args = [token.lower() for token in tokens[1:]]
        if len(args) > 2 or (args and args[-1] not in CONTROL_MODES):
            raise ValueError("usage: mode | mode <force|position> | "
                             "mode <finger> <force|position>")
        with fc.io_lock:
            if args:
                value = CONTROL_MODES.index(args[-1])
                if len(args) == 2:
                    modes = list(fc.hand.get_control_modes())
                    modes[fc._to_index(args[0])] = value
                else:
                    modes = [value] * 6
                fc.hand.set_control_modes(modes)
            # Read back rather than echo, so the reply shows what the hand took.
            modes = fc.hand.get_control_modes()
            fc.control_mode_unreadable = False
        return {name: CONTROL_MODES[1 if value else 0]
                for name, value in zip(FINGER_NAMES, modes)}

    def _force_command(self, tokens):
        """Set ONE finger's force by writing only that finger's register.

        The base command reads all six set forces and writes all six back,
        which hands the other five fingers a new force command as well.
        """
        fc = self.fc
        index = fc._to_index(tokens[1])
        try:
            value = int(tokens[2])
        except ValueError:
            raise ValueError("force must be a whole number of grams") from None
        if not 0 <= value <= 3000:
            raise ValueError("force limit must be 0-3000 g")
        with fc.io_lock:
            fc.hand.set_force(index, value)
        return (f"{FINGER_NAMES[index]} set force = {value} g "
                "(only this finger written)")

    def _forceset_command(self, tokens):
        """Write all six set forces (grams) with a single FORCE_SET write."""
        if len(tokens) != 7:
            raise ValueError("usage: forceset <little> <ring> <middle> "
                             "<index> <thumb_bend> <thumb_rot>")
        try:
            values = [int(token) for token in tokens[1:]]
        except ValueError:
            raise ValueError("force values must be whole grams") from None
        if not all(0 <= value <= 3000 for value in values):
            raise ValueError("force must be 0-3000 g")
        with self.fc.io_lock:
            self.fc.hand.set_forces(values)
        return dict(zip(FINGER_NAMES, values))

    def _execute(self, tokens):
        head = tokens[0].lower()
        if (head == "set" and len(tokens) > 1
                and tokens[1].strip().lower() in ("mode", "forceset", "lc",
                                                  "ae", "mic")):
            raise ValueError(f"'{tokens[1].strip().lower()}' is already a "
                             "built-in command")
        if head in self.shortcuts:      # lets a saved control run `mode ...`
            tokens = expand_shortcut(tokens, self.shortcuts)
            head = tokens[0].lower()
        if head == "cam":
            return self._cam_command(tokens)
        if head == "lc":
            return self._loadcell_command(tokens)
        if head == "ae" and self.ae is not None:
            return self._ae_command(tokens)
        if head == "mic":
            return self._mic_command(tokens)
        if head == "mode":
            return self._mode_command(tokens)
        if head == "forceset":
            return self._forceset_command(tokens)
        if head == "force" and len(tokens) == 3:
            return self._force_command(tokens)
        is_log = (head == "log" and len(tokens) > 1
                  and tokens[1].lower() in ("start", "stop"))
        if not is_log:
            return super()._execute(tokens)
        sub = tokens[1].lower()
        if sub == "stop":
            # video ends with the log; the load cell is detached inside
            # logger.stop() (see hook_logger)
            video = (self.camera.stop_recording() if self.camera.recording
                     else None)
            audio = self.mic.stop_recording()
            result = super()._execute(tokens)
            if isinstance(result, dict):
                if video:
                    result["video"] = video
                if audio:
                    result["audio"] = audio
                if self.loadcell.last_log:
                    result.update(self.loadcell.last_log)
                if self.ae is not None and self.ae.last_log:
                    result.update(self.ae.last_log)
            return result
        result = {"log_path": str(super()._execute(tokens)),
                  "loadcell": ("recording" if self.loadcell.recording
                               else self.loadcell.message)}
        if self.ae is not None:
            result["ae"] = ("recording" if self.ae.recording
                            else self.ae.message)
        if self.camera.available:
            try:
                result["video_path"] = self.camera.start_recording(
                    Path(self.fc.logger.session_dir) / "video")
            except Exception as exc:     # log keeps running without video
                result["video_error"] = str(exc)
        if self.mic.enabled and not self.mic.recording:
            try:
                result["audio_path"] = self.mic.start_recording(
                    Path(self.fc.logger.session_dir) / "audio")
            except Exception as exc:     # log keeps running without audio
                result["audio_error"] = str(exc)
        return result


CAMERA_PANEL = """<section class="panel span12"><h2>Webcam (synced with log)</h2><div class="row" style="align-items:flex-start"><img id="camView" alt="no camera" style="width:320px;max-width:100%;border-radius:10px;background:#07101c;min-height:120px"><div><div id="camState" class="notice">Camera status…</div><div class="toolbar" style="margin-top:10px"><button class="primary" onclick="cmd('cam start')">Record video</button><button onclick="cmd('cam stop')">Stop video</button></div><div class="small" style="margin-top:8px">Starting a log also records video; stopping the log stops it. Frame times are saved in the .frames.csv beside the video.</div></div></div></section>
"""

CAMERA_SCRIPT = """
async function refreshCam(){try{const c=await api('/api/camera');$('camState').innerHTML='<span class="statusdot" style="background:'+(c.available?(c.recording?'var(--orange)':'var(--cyan)'):'var(--red)')+'"></span>'+c.message+(c.recording?' · REC '+c.video_path:'');if(c.available&&!$('camView').src)$('camView').src='/camera.mjpg'}catch(e){}}
refreshCam();setInterval(refreshCam,2000);
"""

FORCE_PANEL = """<section class="panel span12"><h2>Force control</h2><div class="toolbar"><button onclick="cmd('mode position')">All to position mode</button><button class="warn" onclick="cmd('mode force')">All to force mode</button><span id="modeState" class="small">Reading control mode…</span></div><div id="forceFingers" class="fingers"></div><div class="toolbar" style="margin-top:12px"><button class="primary" onclick="applyForces()">Apply all six</button><input id="forceAll" type="number" min="0" max="3000" value="500" aria-label="Set force for all fingers" style="width:90px;background:#091321;border:1px solid var(--line);color:white;padding:9px;border-radius:8px"><button onclick="cmd('forceall '+$('forceAll').value)">Set all to this value</button><span class="small">Set force is in grams, 0–3000.</span></div></section>
"""

MODE_SCRIPT = """
const forceSeen={};
function buildForcePanel(fingers){$('forceFingers').innerHTML=fingers.map(f=>`<div style="border:1px solid var(--line);border-radius:12px;padding:12px;background:#0c1728"><div class="finger-head"><span>${f.name.replace('_',' ')}</span><span class="metric" id="fa-${f.name}"></span></div><div class="angle-row"><input type="range" min="0" max="3000" step="10" id="fr-${f.name}" oninput="$('fn-${f.name}').value=this.value"><input type="number" min="0" max="3000" id="fn-${f.name}" oninput="$('fr-${f.name}').value=this.value"></div><div class="row"><button onclick="cmd('force ${f.name} '+$('fn-${f.name}').value)">Set force</button><button id="fm-${f.name}" onclick="toggleMode('${f.name}')"></button><span class="small" id="fs-${f.name}"></span></div></div>`).join('')}
function toggleMode(name){const f=state.fingers.find(x=>x.name===name);cmd('mode '+name+' '+(f.control_mode==='force'?'position':'force'))}
function renderForcePanel(fingers){if(!$('fr-'+fingers[0].name))buildForcePanel(fingers);fingers.forEach(f=>{$('fa-'+f.name).textContent='actual '+f.force+' g';$('fs-'+f.name).textContent='set '+f.force_set+' g'+(f.control_mode?' · '+f.control_mode:'');const toggle=$('fm-'+f.name),forced=f.control_mode==='force';toggle.style.display=f.control_mode?'':'none';toggle.textContent=forced?'Switch to position':'Switch to force';toggle.className=forced?'':'warn';if(forceSeen[f.name]!==f.force_set){forceSeen[f.name]=f.force_set;$('fr-'+f.name).value=$('fn-'+f.name).value=f.force_set}})}
function applyForces(){cmd('forceset '+state.fingers.map(f=>$('fn-'+f.name).value).join(' '))}
function renderControlMode(fingers){const box=$('modeState'),forced=fingers.filter(f=>f.control_mode==='force').map(f=>f.name.replace('_',' '));if(fingers[0].control_mode==null){box.textContent='Control mode is not reported by this hand.';return}const text=!forced.length?'All fingers in position mode':forced.length===fingers.length?'All fingers in force mode':'Force mode: '+forced.join(', ')+' · others in position mode';box.innerHTML='<span class="statusdot" style="background:'+(forced.length?'var(--orange)':'var(--cyan)')+'"></span>'+text;fingers.forEach(f=>{const slider=$('r-'+f.name);if(slider)slider.closest('.finger').querySelector('.small').append(' · '+f.control_mode)})}
const baseRenderState=renderState;renderState=function(s){baseRenderState(s);renderControlMode(s.fingers);renderForcePanel(s.fingers)};
"""

LOADCELL_PANEL = """<section class="panel span12"><h2>Load cell (HX711 on guide Pico, synced with log)</h2><div class="row" style="align-items:flex-start"><canvas id="lcPlot" style="flex:1 1 520px;min-width:0;height:220px;background:#07101c;border-radius:10px"></canvas><div style="flex:0 1 300px"><div id="lcValue" class="metric" style="font-size:28px;font-weight:700">—</div><div id="lcState" class="notice" style="margin-top:6px">Load cell status…</div><div class="toolbar" style="margin-top:10px"><button onclick="cmd('lc tare')">Tare</button><input id="lcCalG" type="number" min="1" value="500" aria-label="Calibration weight in grams" style="width:90px;background:#091321;border:1px solid var(--line);color:white;padding:9px;border-radius:8px"><button class="primary" onclick="cmd('lc calibrate '+$('lcCalG').value)">Calibrate</button></div><div class="small" style="margin-top:8px">Tare with nothing on the cell, then put the known weight (g) on it and press Calibrate. A running log also stores the load cell in loadcell/ of the same .h5.</div></div></div></section>
"""

LOADCELL_SCRIPT = """
const lc={seq:0,ver:null,pts:[],units:'g',busy:false};
function drawLc(){const c=$('lcPlot'),dpr=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w)return;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);g.font='11px Segoe UI,sans-serif';g.fillStyle='#91a4bd';const P=lc.pts,L=52,R=10,T=10,B=22;if(P.length<2){g.fillText(lc.why||'waiting for load cell samples…',L,h/2);return}const tEnd=P[P.length-1][1],t0=tEnd-10;let lo=Infinity,hi=-Infinity;P.forEach(p=>{if(p[2]<lo)lo=p[2];if(p[2]>hi)hi=p[2]});if(hi-lo<2){hi+=1;lo-=1}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;const x=t=>L+(t-t0)/10*(w-L-R),y=v=>T+(hi-v)/(hi-lo)*(h-T-B);g.strokeStyle='#293b57';g.lineWidth=1;for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,yy=y(v);g.beginPath();g.moveTo(L,yy);g.lineTo(w-R,yy);g.stroke();g.fillText(v.toFixed(hi-lo<20?1:0),4,yy+4)}g.fillText(lc.units,4,T+2);for(let s=0;s<=10;s+=2)g.fillText((s-10)+' s',x(t0+s)-12,h-6);g.strokeStyle='#31d7c8';g.lineWidth=1.6;g.beginPath();P.forEach((p,i)=>{const xx=x(p[1]),yy=y(p[2]);i?g.lineTo(xx,yy):g.moveTo(xx,yy)});g.stroke()}
async function refreshLc(){if(lc.busy)return;lc.busy=true;try{const d=await api('/api/loadcell?since='+lc.seq),s=d.state;if(lc.ver!==null&&d.cal_version!==lc.ver){lc.ver=d.cal_version;lc.pts=[];lc.seq=0;return}lc.ver=d.cal_version;lc.units=d.units;lc.why=s.available?'':'No load cell: '+s.message;d.samples.forEach(p=>lc.pts.push(p));if(d.samples.length)lc.seq=d.samples[d.samples.length-1][0];if(lc.pts.length){const end=lc.pts[lc.pts.length-1][1];while(lc.pts[0][1]<end-10)lc.pts.shift()}$('lcValue').textContent=s.grams!=null?s.grams.toFixed(1)+' g':(s.raw!=null?s.raw+' raw':'—');$('lcState').innerHTML='<span class="statusdot" style="background:'+(s.available?(s.recording?'var(--orange)':'var(--cyan)'):'var(--red)')+'"></span>'+s.message+(s.rate_hz?' · '+s.rate_hz+' Hz (RATE pin '+s.rate_pin+')':'')+(s.recording?' · REC':'')+(s.link_busy?' · guide moving, buffering on Pico':'')+(s.available&&!s.calibrated?' · not calibrated: raw counts':'')+(s.dropped?' · '+s.dropped+' dropped':'');drawLc()}catch(e){}finally{lc.busy=false}}
refreshLc();setInterval(refreshLc,200);window.addEventListener('resize',drawLc);
"""

AE_PANEL = """<section class="panel span12"><h2>Acoustic emission (Board 2, AE-ENV24A10 envelope, synced with log)</h2><div class="row" style="align-items:flex-start"><canvas id="aePlot" style="flex:1 1 520px;min-width:0;height:220px;background:#07101c;border-radius:10px"></canvas><div style="flex:0 1 300px"><div id="aeValue" class="metric" style="font-size:20px;font-weight:700">—</div><div id="aeState" class="notice" style="margin-top:6px">AE status…</div><div class="toolbar" style="margin-top:10px"><button class="primary" onclick="cmd('ae connect ram')">Connect (RAM sampler)</button><button onclick="cmd('ae connect legacy')">Connect (legacy main.py)</button><button onclick="cmd('ae disconnect')">Disconnect</button></div><div class="small" style="margin-top:8px">Live view: RMS per 20 ms (solid) and peak (faint), mV at the ADC pin. Full 20 kHz data goes only to ae/ in the log's .h5. Nothing is ever written to Board 2: disconnecting, or pressing reset on Board 2, returns it to its old AE firmware (main.py), unchanged.</div></div></div></section>
"""

AE_SCRIPT = """
const ae={seq:0,pts:[],busy:false,why:''};
function drawAe(){const c=$('aePlot'),dpr=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w)return;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);g.font='11px Segoe UI,sans-serif';g.fillStyle='#91a4bd';const P=ae.pts,L=52,R=10,T=10,B=22;if(P.length<2){g.fillText(ae.why||'waiting for AE samples…',L,h/2);return}const tEnd=P[P.length-1][1],t0=tEnd-10;let hi=1;P.forEach(p=>{hi=Math.max(hi,p[4],p[5])});hi*=1.08;const x=t=>L+(t-t0)/10*(w-L-R),y=v=>T+(hi-v)/hi*(h-T-B);g.strokeStyle='#293b57';g.lineWidth=1;for(let i=0;i<=4;i++){const v=hi*i/4,yy=y(v);g.beginPath();g.moveTo(L,yy);g.lineTo(w-R,yy);g.stroke();g.fillText(v.toFixed(hi<20?1:0),4,yy+4)}g.fillText('mV',4,T+2);for(let s=0;s<=10;s+=2)g.fillText((s-10)+' s',x(t0+s)-12,h-6);[[2,4,'#31d7c8'],[3,5,'#ff9f43']].forEach(([k,m,col])=>{g.globalAlpha=0.3;g.strokeStyle=col;g.lineWidth=1;g.beginPath();P.forEach((p,i)=>{i?g.lineTo(x(p[1]),y(p[m])):g.moveTo(x(p[1]),y(p[m]))});g.stroke();g.globalAlpha=1;g.lineWidth=1.6;g.beginPath();P.forEach((p,i)=>{i?g.lineTo(x(p[1]),y(p[k])):g.moveTo(x(p[1]),y(p[k]))});g.stroke()});g.fillStyle='#31d7c8';g.fillText('ch0 GP26',w-150,T+10);g.fillStyle='#ff9f43';g.fillText('ch1 GP27',w-80,T+10)}
async function refreshAe(){if(ae.busy)return;ae.busy=true;try{const d=await api('/api/ae?since='+ae.seq),s=d.state;ae.why=s.available?'':'No AE: '+s.message;d.points.forEach(p=>ae.pts.push(p));if(d.points.length)ae.seq=d.points[d.points.length-1][0];if(ae.pts.length){const end=ae.pts[ae.pts.length-1][1];while(ae.pts[0][1]<end-10)ae.pts.shift()}$('aeValue').textContent=s.rms_mv?'RMS '+s.rms_mv[0].toFixed(1)+' / '+s.rms_mv[1].toFixed(1)+' mV':'—';$('aeState').innerHTML='<span class="statusdot" style="background:'+(s.available?(s.recording?'var(--orange)':'var(--cyan)'):'var(--red)')+'"></span>'+s.message+(s.rate_hz?' · '+s.rate_hz+' Hz':'')+(s.recording?' · REC':'')+(s.available?' · '+s.frames+' frames, '+s.dropped+' dropped, '+s.bad+' bad, '+s.overruns+' overruns':'');drawAe()}catch(e){}finally{ae.busy=false}}
refreshAe();setInterval(refreshAe,250);window.addEventListener('resize',drawAe);
"""

MIC_PANEL = """<section class="panel span12"><h2>Microphone (optional, synced with log)</h2><div class="row" style="align-items:flex-start"><div style="flex:1 1 420px"><label class="small" style="display:flex;gap:8px;align-items:center;font-size:14px"><input id="micOn" type="checkbox" onchange="cmd(this.checked?'mic on':'mic off')"> Record microphone with every log</label><div class="toolbar" style="margin-top:10px"><select id="micDev" aria-label="Microphone device" onchange="cmd('mic device '+this.value)" style="max-width:420px;background:#091321;border:1px solid var(--line);color:white;padding:9px;border-radius:8px"><option value="default">System default input</option></select><button class="primary" onclick="cmd('mic start')">Record audio</button><button onclick="cmd('mic stop')">Stop audio</button></div><div class="small" style="margin-top:8px">When ticked, starting a log also records audio.wav (16-bit, 48 kHz) into the session folder and stopping the log stops it. Block times are saved in audio.blocks.csv on the same clock as the .h5. The microphone is only opened while recording.</div></div><div style="flex:0 1 300px"><div id="micLevel" class="metric" style="font-size:20px;font-weight:700">—</div><div id="micState" class="notice" style="margin-top:6px">Microphone status…</div></div></div></section>
"""

MIC_SCRIPT = """
async function loadMicDevices(){try{const list=await api('/api/mic/devices'),sel=$('micDev');list.forEach(d=>{const o=document.createElement('option');o.value=d.index;o.textContent=d.index+': '+d.name+' ('+d.hostapi+')';sel.append(o)})}catch(e){}}
async function refreshMic(){try{const m=await api('/api/mic');$('micOn').checked=m.enabled;const sel=$('micDev'),want=m.device==null?'default':String(m.device);if(document.activeElement!==sel&&[...sel.options].some(o=>o.value===want))sel.value=want;$('micLevel').textContent=m.level_dbfs!=null?m.level_dbfs.toFixed(1)+' dBFS':'—';$('micState').innerHTML='<span class="statusdot" style="background:'+(m.available?(m.recording?'var(--orange)':(m.enabled?'var(--cyan)':'#55657a')):'var(--red)')+'"></span>'+m.message+(m.enabled?' · records with log':' · off')+(m.recording?' · REC '+m.seconds+' s → '+m.audio_path:'')+(m.overflows?' · '+m.overflows+' overflows':'')+(m.error?' · error: '+m.error:'')}catch(e){}}
loadMicDevices().then(refreshMic);setInterval(refreshMic,500);
"""

DATA_LOGGING_PANEL = '<section class="panel span4"><h2>Data logging</h2>'

HTML = HTML.replace(
    '<section class="panel span12"><h2>Command console',
    CAMERA_PANEL + MIC_PANEL + LOADCELL_PANEL + AE_PANEL
    + '<section class="panel span12"><h2>Command console', 1
).replace(DATA_LOGGING_PANEL, FORCE_PANEL + DATA_LOGGING_PANEL, 1
).replace("</script></body>",
          CAMERA_SCRIPT + MIC_SCRIPT + MODE_SCRIPT + LOADCELL_SCRIPT
          + AE_SCRIPT + "</script></body>", 1)


def make_cam_handler(runner, workspace, camera, loadcell, ae=None):
    mic = runner.mic
    Parent = make_handler(runner, workspace)

    class Handler(Parent):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/camera":
                self.json_response(camera.state())
            elif path == "/api/mic":
                self.json_response(mic.state())
            elif path == "/api/mic/devices":
                try:
                    self.json_response(mic.devices())
                except Exception as exc:
                    self.json_response({"error": str(exc)}, 500)
            elif path == "/api/loadcell":
                try:
                    since = int(parse_qs(parsed.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                self.json_response(loadcell.web_payload(since))
            elif path == "/api/ae" and ae is not None:
                try:
                    since = int(parse_qs(parsed.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                self.json_response(ae.web_payload(since))
            elif path == "/camera.mjpg":
                self.stream_camera()
            else:
                super().do_GET()

        def stream_camera(self):
            if not camera.available:
                self.json_response({"error": camera.message}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = camera.jpeg()
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame)).encode()
                            + b"\r\n\r\n" + frame + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionError, OSError):
                pass

    return Handler


def trace_writes(transport):
    """Print every register write, so the terminal shows what reaches the hand."""
    client = transport.client
    for method in ("write_registers", "write_register"):
        def traced(address, values, _send=getattr(client, method), **kwargs):
            shown = list(values) if isinstance(values, (list, tuple)) else values
            # command output is captured for the web reply, so bypass it
            print(f"  [hand] write {address} <- {shown}", file=sys.__stdout__)
            return _send(address, values, **kwargs)
        setattr(client, method, traced)


def build_controller(args):
    """Real hand if reachable; otherwise the simulated hand."""
    if not args.demo:
        try:
            hand = RH56Hand(RH56ModbusTCP(args.robot_ip, port=args.robot_port))
            if getattr(args, "trace_writes", False):
                trace_writes(hand.t)
            fc = FingerController(hand)
            fc.hand.get_angles()            # fail now, not on first poll
            return fc, "real"
        except Exception as exc:
            print(f"Hand not reachable ({exc}); using demo hand.")
    return FingerController(ModeDemoHand()), "demo"


def resolve_guide_port(port):
    """Board 1's COM port. "auto" finds it by USB serial (rig_boards.json);
    an explicit port is used as given, except that Board 2 is refused --
    both boards are 2e8a:0005 and COM numbers swap after replugging."""
    try:
        import ae_board2
        config = ae_board2.load_config()
    except Exception:
        config = None
    if port and port.lower() == "auto":
        port = ae_board2.find_board("board1", config) if config else None
        if port is None:
            print("Board 1 (guide) not found by USB serial; see rig_boards.json")
        return port
    if port and config:
        serial_no = ae_board2.port_serial(port)
        if serial_no and serial_no == (config.get("board2") or {}).get(
                "usb_serial"):
            print(f"{port} is Board 2 (AE), not the guide: not opening it.")
            return None
    return port


def build_guide(args):
    port = resolve_guide_port(args.guide_port)
    if port:
        try:
            return SerialLinearGuide(port, args.guide_baud,
                                     args.guide_timeout)
        except Exception as exc:
            print(f"Linear guide on {port} unavailable ({exc}).")
    return DemoLinearGuide() if args.demo else DisconnectedLinearGuide()


def build_ae(args, logger):
    """Board 2 AE sensors if found by USB serial; the app runs without."""
    ae = AESensor(args.ae_mode, args.ae_port)
    ae._logger = logger
    if args.ae_mode == "off":
        ae.message = "AE disabled (--ae-mode off)"
    elif not ae.open():
        print(f"AE unavailable: {ae.message}")
    return ae


def build_loadcell(args, guide):
    """HX711 on the guide Pico if it answers; simulated with --demo."""
    loadcell = LoadCell(guide, demo=args.demo)
    if args.no_loadcell:
        loadcell.message = "load cell disabled (--no-loadcell)"
    elif not loadcell.open():
        print(f"Load cell unavailable: {loadcell.message}")
    return loadcell


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--demo", action="store_true",
                        help="simulate the hand (and guide if no port)")
    parser.add_argument("--robot-ip", default="192.168.11.210")
    parser.add_argument("--robot-port", type=int, default=6000)
    parser.add_argument("--trace-writes", action="store_true",
                        help="print every register write sent to the hand")
    parser.add_argument("--guide-port", default="auto",
                        help='Board 1 serial port; "auto" = by USB serial '
                             'in rig_boards.json; "" to skip')
    parser.add_argument("--guide-baud", type=int, default=115200)
    parser.add_argument("--guide-timeout", type=float, default=120.0)
    parser.add_argument("--camera", type=int, default=1,
                        help="webcam index (1 = Logitech C930e on this PC, 0 = notebook)")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=float, default=30.0)
    parser.add_argument("--no-loadcell", action="store_true",
                        help="do not start the HX711 load cell on the Pico "
                             "(also stops guide position logging)")
    parser.add_argument("--ae-mode", default="ram",
                        choices=("ram", "legacy", "off"),
                        help="Board 2 AE: ram = RAM sampler (3.2 ms frames), "
                             "legacy = untouched main.py stream, off")
    parser.add_argument("--ae-port", default=None,
                        help="override Board 2's port (still checked by "
                             "USB serial and file list)")
    parser.add_argument("--mic", action="store_true",
                        help="record the microphone with every log "
                             "(also a checkbox in the page / `mic on`)")
    parser.add_argument("--mic-device", default=None,
                        help="input device index or part of its name "
                             "(default: system default input; `mic devices`)")
    parser.add_argument("--mic-rate", type=int, default=48000)
    parser.add_argument("--mic-channels", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)

    fc, mode = build_controller(args)
    guide = build_guide(args)
    camera = Webcam(args.camera, args.cam_width, args.cam_height, args.cam_fps)
    if args.no_camera:
        camera.message = "camera disabled (--no-camera)"
    elif not camera.open():
        print(f"Webcam unavailable: {camera.message}")

    loadcell = build_loadcell(args, guide)
    ae = build_ae(args, fc.logger)
    hook_logger(fc.logger, loadcell, ae)

    device = args.mic_device
    mic = Microphone(int(device) if device and device.isdigit() else device,
                     args.mic_rate, args.mic_channels, enabled=args.mic)
    if not mic.probe() and args.mic:
        print(f"Microphone unavailable: {mic.message}")

    runner = CamCommandRunner(fc, mode, guide, camera, loadcell, ae, mic)
    workspace = Path.cwd().resolve()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_cam_handler(runner, workspace, camera, loadcell, ae))
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"RH56 web dashboard: {url}")
    print(f"Hand: {mode.upper()} | Guide: {guide.mode.upper()} | "
          f"Camera: {camera.message} | Load cell: {loadcell.message}")
    print(f"AE: {ae.message}")
    print(f"Microphone: {mic.message} "
          f"({'records with log' if mic.enabled else 'off; --mic or page checkbox'})")
    print("Press Ctrl+C to stop.")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web dashboard.")
    finally:
        server.server_close()
        runner.routine.stop()
        camera.stop_recording()
        mic.close()
        fc.stop_log()               # drains the load cell before closing
        camera.close()
        ae.close()                  # Board 2 back to its untouched main.py
        loadcell.close()            # stops core 1 before the guide lets go
        fc.hand.t.close()
        guide.close()


if __name__ == "__main__":
    main()
