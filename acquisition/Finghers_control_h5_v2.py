"""
Per-finger control for the RH56E2 hand
======================================
Uses the -1 convention of ANGLE_SET: any DOF set to -1 takes no action,
so we can command one finger while all others hold position.

Run interactively:
    python finger_control.py

Or import:
    from rh56_hand import RH56Hand, RH56ModbusTCP
    from finger_control import FingerController
    fc = FingerController(RH56Hand(RH56ModbusTCP("192.168.11.210", port=6000)))
    fc.move("index", 300)          # bend index finger to 300
    fc.move(3, 1000)               # same finger by DOF index, fully open
"""

import csv
import itertools
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path
from H1 import (RH56Hand, RH56ModbusTCP, RH56Serial,
                       STATUS_TEXT, FINGER_NAMES)

# name -> DOF index (aliases included)
FINGERS = {
    "little": 0, "pinky": 0,
    "ring": 1,
    "middle": 2,
    "index": 3,
    "thumb": 4, "thumb_bend": 4,
    "thumb_rot": 5, "thumb_rotation": 5,
}

# Preferred display names for the six DOFs.  In particular, keep the shorter
# "thumb" name in generated experiment sequences instead of "thumb_bend".
CANONICAL_FINGERS = ("little", "ring", "middle", "index", "thumb",
                     "thumb_rot")

# RH56E2/DFTP tactile blocks from the bundled user manual and tactile SDK:
# (byte address, byte length, displayed rows, displayed columns).  This map is
# intentionally local so all tactile logging/plotting behavior lives in this
# finger-control module.
TACTILE = {
    "little_tip":  (3000, 18, 3, 3),   "little_nail": (3018, 192, 12, 8),
    "little_pad":  (3210, 160, 10, 8), "ring_tip":    (3370, 18, 3, 3),
    "ring_nail":   (3388, 192, 12, 8), "ring_pad":    (3580, 160, 10, 8),
    "middle_tip":  (3740, 18, 3, 3),   "middle_nail": (3758, 192, 12, 8),
    "middle_pad":  (3950, 160, 10, 8), "index_tip":   (4110, 18, 3, 3),
    "index_nail":  (4128, 192, 12, 8), "index_pad":   (4320, 160, 10, 8),
    "thumb_tip":   (4480, 18, 3, 3),   "thumb_nail":  (4498, 192, 12, 8),
    "thumb_mid":   (4690, 18, 3, 3),   "thumb_pad":   (4708, 192, 12, 8),
    "palm":        (4900, 224, 8, 14),
}

# Storage layout.  TACTILE above keeps the *hardware* register names required by
# read_tactile(); this map translates each one into the anatomical name used in
# the HDF5 file, so the file reads like the hand rather than like the protocol.
#   hardware region -> (finger group, dataset name)
TACTILE_LAYOUT = {
    "thumb_tip":   ("thumb",  "fingertip"),
    "thumb_nail":  ("thumb",  "distal_phalanx"),
    "thumb_mid":   ("thumb",  "proximal_phalanx"),
    "index_tip":   ("index",  "fingertip"),
    "index_nail":  ("index",  "distal_phalanx"),
    "index_pad":   ("index",  "middle_phalanx"),
    "middle_tip":  ("middle", "fingertip"),
    "middle_nail": ("middle", "distal_phalanx"),
    "middle_pad":  ("middle", "middle_phalanx"),
    "ring_tip":    ("ring",   "fingertip"),
    "ring_nail":   ("ring",   "distal_phalanx"),
    "ring_pad":    ("ring",   "middle_phalanx"),
    "little_tip":  ("little", "fingertip"),
    "little_nail": ("little", "distal_phalanx"),
    "little_pad":  ("little", "middle_phalanx"),
    # thumb_pad sits on the palm at the base of the thumb: it is the thenar
    # eminence, not a pad on the thumb itself.
    "thumb_pad":   ("thumb",   "thenar"),
    "palm":        ("palm",   "palm"),
}

# DOF index -> (finger group, actuator sub-group).  The thumb owns two DOFs, so
# its motor channels live one level deeper to keep their names unambiguous.
DOF_LAYOUT = {
    0: ("little", None),
    1: ("ring",   None),
    2: ("middle", None),
    3: ("index",  None),
    4: ("thumb",  "bend"),
    5: ("thumb",  "rot"),
}

# Finger groups in anatomical order; palm carries tactile pads but no actuator.
HAND_GROUPS = ("thumb", "index", "middle", "ring", "little", "palm")


def tactile_path(region, root=""):
    """HDF5 path of a hardware tactile region, e.g. 'index/tactile/fingertip'."""
    group, name = TACTILE_LAYOUT[region]
    return f"{root}{group}/tactile/{name}"


def tactile_label(region):
    """Human-readable name of a region, e.g. 'index fingertip'."""
    group, name = TACTILE_LAYOUT[region]
    return f"{group} {name}".replace("_", " ")


def dof_path(index, root=""):
    """HDF5 group holding one DOF's motor channels, e.g. 'thumb/bend'."""
    group, sub_group = DOF_LAYOUT[index]
    return f"{root}{group}" if sub_group is None else f"{root}{group}/{sub_group}"


def iso_local(unix_time):
    """Local ISO-8601 timestamp with milliseconds and UTC offset.

    ``2026-08-24T10:47:03.412+0700`` — unambiguous even when the file is
    opened in another timezone, unlike a bare date/time string.
    """
    local = time.localtime(unix_time)
    milliseconds = int(round((unix_time % 1) * 1000)) % 1000
    return (time.strftime("%Y-%m-%dT%H:%M:%S", local)
            + f".{milliseconds:03d}" + time.strftime("%z", local))


def _write_time_attrs(target, unix_time, prefix="created"):
    """Store one instant as unix seconds plus readable date/time fields."""
    local = time.localtime(unix_time)
    target.attrs[f"{prefix}_unix"] = float(unix_time)
    target.attrs[f"{prefix}_iso"] = iso_local(unix_time)
    target.attrs[f"{prefix}_date"] = time.strftime("%Y-%m-%d", local)
    target.attrs[f"{prefix}_time"] = time.strftime("%H:%M:%S", local)
    target.attrs[f"{prefix}_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(unix_time))


def _require_h5py():
    """Import HDF5 support with one actionable dependency message."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "HDF5 logging requires h5py and numpy. Install them with: "
            "pip install h5py numpy"
        ) from exc
    return h5py, np


def generate_grasp_sequences(first_group=("index", "thumb"),
                             remaining=("middle", "ring", "little"),
                             randomize_trials=False, seed=None):
    """Generate every grasp order while keeping the first stage fixed.

    Each returned sequence is a tuple of *stages*.  Fingers in the same stage
    are intended to move together, so the default first stage makes the index
    finger and thumb grasp simultaneously.  The remaining fingers each have
    their own sequential stage.

    With the defaults this returns all 3! = 6 experimental conditions::

        (("index", "thumb"), ("middle",), ("ring",), ("little",))
        (("index", "thumb"), ("middle",), ("little",), ("ring",))
        ...

    Set ``randomize_trials=True`` to shuffle the order in which the conditions
    are presented.  Supplying ``seed`` makes that shuffle reproducible; it
    does not remove or duplicate any condition.
    """
    def normalize_group(group, argument):
        if isinstance(group, str):
            group = (group,)
        normalized = []
        for finger in group:
            key = str(finger).strip().lower()
            if key not in FINGERS:
                raise ValueError(
                    f"unknown finger '{finger}' in {argument} "
                    f"(use {sorted(set(FINGERS))})"
                )
            name = CANONICAL_FINGERS[FINGERS[key]]
            if name in normalized:
                raise ValueError(f"duplicate finger '{name}' in {argument}")
            normalized.append(name)
        return tuple(normalized)

    fixed = normalize_group(first_group, "first_group")
    others = normalize_group(remaining, "remaining")
    overlap = set(fixed).intersection(others)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"finger(s) present in both groups: {names}")
    if not fixed:
        raise ValueError("first_group must contain at least one finger")

    sequences = [
        (fixed,) + tuple((finger,) for finger in order)
        for order in itertools.permutations(others)
    ]
    if randomize_trials:
        random.Random(seed).shuffle(sequences)
    return sequences


class ParamLogger:
    """Background HDF5 logger for hand parameters and tactile sensor pads.

    Samples all six DOFs at a fixed rate in its own thread, so a running
    sequence is captured without the caller having to poll. Every read goes
    through the controller's io_lock so it never collides with a move command
    on the same (non-thread-safe) transport.

    The file is grouped by finger.  Every motor channel is a ``(sample,)``
    dataset inside its finger's group, and every tactile pad is a
    ``(tactile_sample, row, column)`` dataset under ``<finger>/tactile/``,
    keeping its physical matrix shape and its lower-rate timeline::

        time/timestamp, time/elapsed_s
        time/tactile_timestamp, time/tactile_elapsed_s
        index/angle, index/pos, index/force_g, ...
        index/tactile/fingertip, index/tactile/distal_phalanx, ...
        thumb/bend/angle, thumb/rot/angle, ...      (thumb has two DOFs)
        palm/tactile/thenar, palm/tactile/palm      (no actuator)
        marks/<finger>/...                          (same shape as above)
    """

    # (HDF5 dataset name, key in FingerController._read_all()).
    # Covers every read-only measured register in the RH56 manual (2.4.14-20).
    COLUMNS = [
        ("angle", "angles"),        # ANGLE_ACT  0..1000
        ("pos", "positions"),       # POS_ACT    0..2000
        ("force_g", "forces"),      # FORCE_ACT  grams
        ("speed", "speeds"),        # SPEED_SET  0..1000 (setpoint)
        ("current_mA", "currents"),  # CURRENT    0..1000 mA
        ("temp_C", "temps"),        # TEMP       0..100 C
        ("status", "status"),       # STATUS     0..7
        ("error", "errors"),        # ERROR      bitfield
    ]

    def __init__(self, controller, rate_hz=20.0, tactile_hz=5.0):
        self.fc = controller
        self.period = 1.0 / rate_hz if rate_hz > 0 else 0.05
        self.tactile_period = (1.0 / tactile_hz
                               if tactile_hz and tactile_hz > 0 else None)
        self.path = None
        self._file = None
        self._thread = None
        self._stop = threading.Event()
        self._t0 = None
        self._last_tactile_time = None
        self.session_dir = None
        self.marks_dir = None
        self._mark_lock = threading.Lock()
        self._mark_count = 0

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, path=None):
        self.stop()  # never leave a previous file/thread dangling
        h5py, _ = _require_h5py()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if path:
            self.path = Path(path)
            if self.path.suffix.lower() not in (".h5", ".hdf5"):
                if self.path.suffix:
                    raise ValueError("log path must use .h5 or .hdf5")
                self.path = self.path.with_suffix(".h5")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.session_dir = (self.path.parent
                                / f"{self.path.stem}_data")
        else:
            self.session_dir = Path("data_logs") / f"session_{stamp}"
            self.path = self.session_dir / "hand.h5"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.marks_dir = self.session_dir / "marks"
        self.marks_dir.mkdir(exist_ok=True)
        self._file = h5py.File(self.path, "w")
        started_at = time.time()
        self._file.attrs["format"] = "RH56_HAND_LOG"
        self._file.attrs["format_version"] = 2
        _write_time_attrs(self._file, started_at, "created")
        self._file.attrs["timezone"] = time.strftime("%Z", time.localtime())
        self._file.attrs["utc_offset_s"] = int(
            -(time.altzone if time.localtime().tm_isdst else time.timezone))
        self._file.attrs["finger_names"] = json.dumps(list(FINGER_NAMES))
        self._file.attrs["layout"] = "per_finger"
        self._file.attrs["hand_groups"] = json.dumps(list(HAND_GROUPS))
        time_group = self._file.create_group("time")
        for name in ("timestamp", "elapsed_s",
                     "tactile_timestamp", "tactile_elapsed_s"):
            dataset = time_group.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
            dataset.attrs["units"] = ("seconds since 1970-01-01 00:00:00 UTC"
                                      if name.endswith("timestamp")
                                      else "seconds since log start")
        # Readable clock time per sample, so the file is legible in HDFView
        # without converting unix seconds by hand.  Text compresses hard, so
        # the cost next to the tactile arrays is negligible.
        for name in ("datetime", "tactile_datetime"):
            dataset = time_group.create_dataset(
                name, shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"), chunks=(256,),
                compression="gzip", compression_opts=4)
            dataset.attrs["format"] = "ISO-8601 local time with UTC offset"
        self._make_hand_groups(self._file, motor_chunk=512, tactile=True)
        marks = self._file.create_group("marks")
        marks.create_dataset("id", shape=(0,), maxshape=(None,), dtype="i4",
                             chunks=True)
        marks.create_dataset("label", shape=(0,), maxshape=(None,),
                             dtype=h5py.string_dtype("utf-8"), chunks=True)
        marks.create_dataset("timestamp", shape=(0,), maxshape=(None,),
                             dtype="f8", chunks=True)
        marks.create_dataset("elapsed_s", shape=(0,), maxshape=(None,),
                             dtype="f8", chunks=True)
        marks.create_dataset("datetime", shape=(0,), maxshape=(None,),
                             dtype=h5py.string_dtype("utf-8"), chunks=True)
        self._make_hand_groups(marks, motor_chunk=32, tactile=True)
        self._stop.clear()
        self._t0 = time.time()
        self._last_tactile_time = None
        self._mark_count = 0
        self._thread = threading.Thread(target=self._loop,
                                        name="param-logger", daemon=True)
        self._thread.start()
        return str(self.path)

    def _make_hand_groups(self, parent, motor_chunk, tactile):
        """Create <finger>/<channel> and <finger>/tactile/<pad> under `parent`."""
        for index in sorted(DOF_LAYOUT):
            group = parent.require_group(dof_path(index))
            for prefix, _ in self.COLUMNS:
                group.create_dataset(
                    prefix, shape=(0,), maxshape=(None,), dtype="i4",
                    chunks=(motor_chunk,), compression="gzip",
                    compression_opts=1,
                )
        if not tactile:
            return
        for region, (_, _, rows, cols) in TACTILE.items():
            group = parent.require_group(f"{TACTILE_LAYOUT[region][0]}/tactile")
            dataset = group.create_dataset(
                TACTILE_LAYOUT[region][1], shape=(0, rows, cols),
                maxshape=(None, rows, cols), dtype="u2",
                chunks=(1, rows, cols), compression="gzip",
                compression_opts=2,
            )
            dataset.attrs["hardware_region"] = region
            dataset.attrs["rows"] = rows
            dataset.attrs["cols"] = cols

    @staticmethod
    def _append(dataset, value):
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                snap = self.fc._read_all()
                tactile_due = (
                    self.tactile_period is not None
                    and (self._last_tactile_time is None
                         or t - self._last_tactile_time >= self.tactile_period)
                )
                tactile = None
                if tactile_due:
                    try:
                        tactile = self.fc._read_tactile_all()
                        self._last_tactile_time = t
                    except Exception as exc:
                        self._last_tactile_time = t
                        print(f"  [logger] tactile read error: {exc}")
                with self._mark_lock:
                    self._append(self._file["time/timestamp"], t)
                    self._append(self._file["time/elapsed_s"], t - self._t0)
                    self._append(self._file["time/datetime"], iso_local(t))
                    for prefix, key in self.COLUMNS:
                        for index in DOF_LAYOUT:
                            self._append(
                                self._file[f"{dof_path(index)}/{prefix}"],
                                snap[key][index])
                    if tactile is not None:
                        self._append(self._file["time/tactile_timestamp"], t)
                        self._append(self._file["time/tactile_elapsed_s"],
                                     t - self._t0)
                        self._append(self._file["time/tactile_datetime"],
                                     iso_local(t))
                        for region, matrix in tactile.items():
                            self._append(self._file[tactile_path(region)],
                                         matrix)
                self._file.flush()
            except Exception as exc:            # keep sampling despite a glitch
                print(f"  [logger] read error: {exc}")
            self._stop.wait(max(0.0, self.period - (time.time() - t)))

    def mark(self, label=None):
        """Store an exact parameter/tactile snapshot in the HDF5 log."""
        if not self.active:
            raise RuntimeError("start logging before using mark")
        label = str(label).strip() if label else "point_of_interest"
        timestamp = time.time()
        elapsed = timestamp - self._t0
        snap = self.fc._read_all()
        tactile = self.fc._read_tactile_all()
        with self._mark_lock:
            self._mark_count += 1
            mark_id = self._mark_count
            self._append(self._file["marks/id"], mark_id)
            self._append(self._file["marks/label"], label)
            self._append(self._file["marks/timestamp"], timestamp)
            self._append(self._file["marks/elapsed_s"], elapsed)
            self._append(self._file["marks/datetime"], iso_local(timestamp))
            for prefix, key in self.COLUMNS:
                for index in DOF_LAYOUT:
                    self._append(
                        self._file[f"marks/{dof_path(index)}/{prefix}"],
                        snap[key][index])
            for region, matrix in tactile.items():
                self._append(self._file[tactile_path(region, "marks/")], matrix)
            self._file.flush()
        safe_label = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in label
        ).strip("_")[:60] or "point"
        stem = f"mark_{mark_id:03d}_{safe_label}"
        plot_path = self.marks_dir / f"{stem}_tactile.png"
        plot_tactile_snapshot(
            tactile, plot_path,
            title=f"Mark {mark_id}: {label} — t={elapsed:.3f} s",
        )
        return {
            "mark_id": mark_id,
            "mark_label": label,
            "timestamp": timestamp,
            "datetime": iso_local(timestamp),
            "elapsed_s": elapsed,
            "snapshot_h5": str(self.path),
            "tactile_plot": str(plot_path),
        }

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._file:
            # Written at close so a reader knows the span without scanning
            # every sample (and can tell a truncated file from a clean one).
            closed_at = time.time()
            _write_time_attrs(self._file, closed_at, "closed")
            if self._t0 is not None:
                self._file.attrs["duration_s"] = float(closed_at - self._t0)
            self._file.attrs["n_samples"] = int(
                self._file["time/elapsed_s"].shape[0])
            self._file.attrs["n_tactile_frames"] = int(
                self._file["time/tactile_elapsed_s"].shape[0])
            self._file.close()
            self._file = None
        return str(self.path) if self.path else None


class FingerController:
    """Single-finger command layer on top of RH56Hand."""

    def __init__(self, hand: RH56Hand, default_speed=500):
        self.hand = hand
        # Fallback speed applied to a DOF whose SPEED_SET is 0 (which would
        # otherwise never move). Does not override an already-set speed.
        self.default_speed = default_speed
        # Serializes every transport access so the background ParamLogger and
        # foreground move/read commands never talk on the socket at once.
        self.io_lock = threading.RLock()
        self.logger = ParamLogger(self)
        # Deliberately not loaded from disk: the drill grip can change every
        # time, so each program run must perform a fresh trigger calibration.
        self.drill_calibration = None
        self.drill_probe = None

    # ---- logging ---------------------------------------------------------
    def _read_all(self):
        """Locked snapshot of every readable parameter for all six DOFs.

        Covers all read-only measured registers in the RH56 manual: actual
        angle, actual actuator position, actual force, speed setpoint, actuator
        current, temperature, status code and error bitfield.
        """
        with self.io_lock:
            return {
                "angles": self.hand.get_angles(),
                "positions": self.hand.t.read_shorts("POS_ACT"),
                "forces": self.hand.get_forces(),
                "speeds": self.hand.t.read_shorts("SPEED_SET"),
                "currents": self.hand.get_currents(),
                "temps": self.hand.get_temperatures(),
                "status": self.hand.get_status(),
                "errors": self.hand.get_errors(),
            }

    def start_log(self, path=None, rate_hz=20.0, tactile_hz=5.0):
        """Start an HDF5 log, including all tactile pads when available.

        Motor/force parameters are sampled at ``rate_hz``.  The much larger
        tactile map is sampled at ``tactile_hz`` and its latest values are
        repeated between updates.  Set ``tactile_hz=0`` to disable tactile.
        """
        self.logger.period = 1.0 / rate_hz if rate_hz > 0 else 0.05
        self.logger.tactile_period = (1.0 / tactile_hz
                                      if tactile_hz and tactile_hz > 0
                                      else None)
        return self.logger.start(path)

    def stop_log(self):
        """Stop logging and close the HDF5 file. Returns its path."""
        return self.logger.stop()

    def mark_log(self, label=None):
        """Mark the current log time and export an exact tactile snapshot."""
        return self.logger.mark(label)

    def plottactile(self, path=None):
        """Read and save the tactile status at this exact moment.

        The tactile sensors are read once. That same frame is written as one
        frame in HDF5 and rendered as a heatmap PNG. ``path`` may be a target
        ``.h5`` file, a filename stem, or an existing output directory. When
        omitted, both files are placed in ``tactile_snapshots``.
        """
        captured_at = time.time()
        tactile = self._read_tactile_all()
        stamp = time.strftime("%Y%m%d_%H%M%S",
                              time.localtime(captured_at))
        milliseconds = int((captured_at % 1) * 1000)
        default_stem = f"tactile_{stamp}_{milliseconds:03d}"

        if path is None:
            h5_path = Path("tactile_snapshots") / f"{default_stem}.h5"
        else:
            requested = Path(path)
            if requested.suffix.lower() in (".h5", ".hdf5"):
                h5_path = requested
            elif requested.suffix:
                raise ValueError(
                    "plottactile path must be a .h5 file, filename stem, "
                    "or output directory"
                )
            elif requested.exists() and requested.is_dir():
                h5_path = requested / f"{default_stem}.h5"
            else:
                h5_path = requested.with_suffix(".h5")
        png_path = h5_path.with_suffix(".png")
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = iso_local(captured_at)
        h5py, np = _require_h5py()
        with h5py.File(h5_path, "w") as file:
            file.attrs["format"] = "RH56_TACTILE_SNAPSHOT"
            file.attrs["format_version"] = 2
            file.attrs["timestamp"] = timestamp
            _write_time_attrs(file, captured_at, "captured")
            file.attrs["timezone"] = time.strftime("%Z",
                                                   time.localtime(captured_at))
            file.attrs["layout"] = "per_finger"
            for region, matrix in tactile.items():
                group_name, name = TACTILE_LAYOUT[region]
                group = file.require_group(f"{group_name}/tactile")
                dataset = group.create_dataset(
                    name, data=np.asarray(matrix, dtype="u2"),
                    compression="gzip", compression_opts=2)
                dataset.attrs["hardware_region"] = region

        title = f"Tactile snapshot — {timestamp}"
        plot_tactile_snapshot(tactile, png_path, title=title)
        return {
            "timestamp": timestamp,
            "h5_path": str(h5_path),
            "png_path": str(png_path),
        }

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _to_index(finger):
        """Accept DOF index (0-5) or finger name."""
        if isinstance(finger, int):
            if not 0 <= finger <= 5:
                raise ValueError("DOF index must be 0-5")
            return finger
        key = str(finger).strip().lower()
        if key not in FINGERS:
            raise ValueError(f"unknown finger '{finger}' "
                             f"(use {sorted(set(FINGERS))} or 0-5)")
        return FINGERS[key]

    def _mask(self, index, value):
        """Build a 6-value list with `value` at index and -1 elsewhere."""
        arr = [-1] * 6
        arr[index] = value
        return arr

    def _read_tactile_all(self):
        """Locked snapshot of all 1,062 RH56E2 tactile sensor cells.

        The manual exposes tactile only on the E2/DFTP Modbus interface.  Each
        value is an unsigned 16-bit raw sensor reading; regions retain their
        documented 2D shape so they can be rendered as heatmaps.
        """
        read_tactile = getattr(self.hand.t, "read_tactile", None)
        if not callable(read_tactile):
            raise RuntimeError(
                "tactile reading requires the RH56E2/DFTP Modbus transport"
            )
        with self.io_lock:
            tactile = {}
            for region in TACTILE:
                matrix = read_tactile(region)
                if region == "palm":
                    # Raw palm order is 14x8; the manual displays its 8x14
                    # transpose.  Flatten first because the base transport's
                    # generic reader may already have chunked it as 8x14.
                    values = [value for row in matrix for value in row]
                    raw = [values[r * 8:(r + 1) * 8] for r in range(14)]
                    matrix = [list(row) for row in zip(*raw)]
                elif region == "thumb_pad":
                    # Rotate into the physical orientation used by the
                    # manufacturer's tactile example.
                    matrix = [row[::-1] for row in matrix[::-1]]
                tactile[region] = matrix
            return tactile

    # ---- single-finger commands ------------------------------------------
    def move(self, finger, angle, speed=None, force=None, wait=False):
        """
        Move ONE finger to `angle` (0-1000). Others hold position.
        Optionally set that finger's speed / force threshold first.
        """
        i = self._to_index(finger)
        if not (0 <= angle <= 1000):
            raise ValueError("angle must be 0-1000")

        with self.io_lock:
            # SPEED_SET has no -1 convention -> read-modify-write.
            # A DOF whose speed is 0 will not move at all, so guarantee a
            # usable value for the finger we are about to command.
            speeds = self.hand.t.read_shorts("SPEED_SET")
            if speed is not None:
                speeds[i] = speed
                self.hand.set_speeds(speeds)
            elif speeds[i] <= 0:
                speeds[i] = self.default_speed
                self.hand.set_speeds(speeds)

            if force is not None:
                forces = self.hand.t.read_shorts("FORCE_SET")
                forces[i] = force
                self.hand.set_forces(forces)

            self.hand.set_angles(self._mask(i, angle))
        if wait:
            return self.wait_finger(i, target=angle)
        return True

    def bend(self, finger, wait=False):
        """Fully bend one finger (angle 0)."""
        return self.move(finger, 0, wait=wait)

    def open(self, finger, wait=False):
        """Fully open one finger (angle 1000)."""
        return self.move(finger, 1000, wait=wait)

    def set_finger_force(self, finger, force_g):
        """Change force-control threshold (grams) of one finger only.

        The finger stops closing once its measured force reaches this limit.
        Range is model dependent (RH56 0-1000, E2/DFTP 0-3000)."""
        i = self._to_index(finger)
        if not (0 <= force_g <= 3000):
            raise ValueError("force limit must be 0-3000 g")
        with self.io_lock:
            forces = self.hand.t.read_shorts("FORCE_SET")
            forces[i] = force_g
            return self.hand.set_forces(forces)

    def set_finger_speed(self, finger, speed):
        """Change movement speed (0-1000) of one finger only, without moving it."""
        i = self._to_index(finger)
        if not (0 <= speed <= 1000):
            raise ValueError("speed must be 0-1000")
        with self.io_lock:
            speeds = self.hand.t.read_shorts("SPEED_SET")
            speeds[i] = speed
            return self.hand.set_speeds(speeds)

    def wait_finger(self, finger, target=None, timeout=5.0, poll=0.02, tol=15):
        """Block until this one finger finishes its current move.

        Returns its final status code.

        Uses the measured angle (ANGLE_ACT) as the primary completion signal.
        Waiting on the status byte alone is unreliable here: right after a
        command the finger still reports the stale 'reached position' (2) code
        from its *previous* move, so a plain status poll returns instantly and
        the next finger in a sequence fires before this one has moved. Watching
        the angle converge on the target (or a hard force/fault stop) avoids
        that race.
        """
        i = self._to_index(finger)
        t0 = time.time()
        last, stable = None, 0
        while time.time() - t0 < timeout:
            with self.io_lock:
                ang = self.hand.get_angles()[i]
                s = self.hand.get_status()[i]
            # hard stops: force threshold / current protection / rotor / fault
            if s in (3, 5, 6, 7):
                return s
            # reached the commanded angle
            if target is not None and abs(ang - target) <= tol:
                return s
            # fallback: angle has settled and firmware reports 'reached'
            if last is not None and abs(ang - last) <= 1:
                stable += 1
                if stable >= 3 and s == 2:
                    return s
            else:
                stable = 0
            last = ang
            time.sleep(poll)
        with self.io_lock:
            return self.hand.get_status()[i]

    def finger_state(self, finger):
        """Snapshot of one finger: angle, force, status, current, temp."""
        i = self._to_index(finger)
        with self.io_lock:
            return {
                "finger": FINGER_NAMES[i],
                "angle": self.hand.get_angles()[i],
                "force_g": self.hand.get_forces()[i],
                "status": STATUS_TEXT.get(self.hand.get_status()[i], "?"),
                "current_mA": self.hand.get_currents()[i],
                "temp_C": self.hand.get_temperatures()[i],
            }

    # ---- multi-finger convenience ------------------------------------------
    def move_fingers(self, mapping, wait=False):
        """
        Move several selected fingers at once, others hold.
        Example: fc.move_fingers({"index": 300, "thumb": 400, "thumb_rot": 200})
        """
        arr = [-1] * 6
        for f, a in mapping.items():
            arr[self._to_index(f)] = a
        with self.io_lock:
            self.hand.set_angles(arr)
        if wait:
            # brief settle so status leaves the stale 'reached' state before
            # we start polling for the new move to finish
            time.sleep(0.1)
            with self.io_lock:
                self.hand.wait_until_stopped()
        return True

    def move_sequence(self, steps, delay=0.0, speed=None, log=False):
        """Move fingers one at a time, each finishing before the next starts.

        This is the reliable way to drive fingers "separately in sequence":
        every step waits for its finger to reach position (or force-stop)
        before the next step is commanded.

        steps: iterable of (finger, angle) or (finger, angle, force).
        log:   if True, records an HDF5 log for the whole sequence (only
               starts/stops the logger if one isn't already running).
        Example:
            fc.move_sequence([("index", 300), ("middle", 300),
                              ("ring", 300), ("little", 300),
                              ("thumb", 400)], delay=0.2, log=True)
        """
        started_here = False
        if log and not self.logger.active:
            self.start_log()
            started_here = True
        try:
            done = []
            for step in steps:
                finger, angle = step[0], step[1]
                force = step[2] if len(step) > 2 else None
                self.move(finger, angle, speed=speed, force=force, wait=True)
                done.append((FINGER_NAMES[self._to_index(finger)], angle))
                if delay:
                    time.sleep(delay)
            return done
        finally:
            if started_here:
                self.stop_log()

    def move_timed_sequence(self, actions, speed=None, log=False):
        """Run finger moves and explicit millisecond delays in order.

        ``actions`` contains ``(finger, angle)`` move actions and
        ``("delay", milliseconds)`` pause actions.  Every move finishes before
        its following delay begins.  This is the programmatic equivalent of::

            seq index 1000 delay 1000 index 500 delay 1000 index 0

        Returns tagged results such as ``("move", "index", 500)`` and
        ``("delay", 1000)`` so callers can report exactly what happened.
        """
        started_here = False
        if log and not self.logger.active:
            self.start_log()
            started_here = True
        try:
            done = []
            for action in actions:
                if len(action) != 2:
                    raise ValueError(
                        "each timed-sequence action must contain two values"
                    )
                name, value = action
                if str(name).strip().lower() == "delay":
                    milliseconds = int(value)
                    if milliseconds < 0:
                        raise ValueError("delay must be 0 or more milliseconds")
                    time.sleep(milliseconds / 1000.0)
                    done.append(("delay", milliseconds))
                else:
                    angle = int(value)
                    self.move(name, angle, speed=speed, wait=True)
                    canonical = CANONICAL_FINGERS[self._to_index(name)]
                    done.append(("move", canonical, angle))
            return done
        finally:
            if started_here:
                self.stop_log()

    def calibrate_drill_trigger(self, start_angle=1000, deepest_command=0,
                                max_force_g=3000, probe_speed=1000,
                                baseline_samples=10, sample_hz=100.0,
                                tactile_hz=5.0, timeout=10.0,
                                start_tolerance=30,
                                log_path=None):
        """Record one continuous index motion from released to command zero.

        This method deliberately does not guess first contact. It requires the
        index to already be released, records a stationary force/tactile
        baseline, sends exactly one movement command (normally angle zero),
        and samples the resulting continuous motion. Use the generated graph
        to select contact, then call :meth:`set_drill_calibration`. Defaults
        use the device maximum speed (1000) and E2 force limit (3000 g).
        Angle, actuator position and force target 100 Hz while the much larger
        tactile blocks are read independently at 5 Hz.

        The successful probe remains at its deepest measured position. A probe
        error releases the index for safety.
        """
        if not (0 <= deepest_command < start_angle <= 1000):
            raise ValueError(
                "require 0 <= deepest_command < start_angle <= 1000"
            )
        if not (0 < max_force_g <= 3000):
            raise ValueError("max_force_g must be between 1 and 3000")
        if baseline_samples < 1:
            raise ValueError("baseline_samples must be at least 1")
        if sample_hz <= 0 or tactile_hz <= 0 or timeout <= 0:
            raise ValueError(
                "sample_hz, tactile_hz and timeout must be greater than 0"
            )

        index = self._to_index("index")
        with self.io_lock:
            initial_angle = self.hand.get_angles()[index]
            original_speeds = self.hand.t.read_shorts("SPEED_SET")
            original_forces = self.hand.t.read_shorts("FORCE_SET")
        if initial_angle < start_angle - start_tolerance:
            raise RuntimeError(
                f"index is at {initial_angle}; run drillrelease first so it "
                f"is near {start_angle} before recording the one-motion probe"
            )

        safe_forces = list(original_forces)
        safe_forces[index] = int(max_force_g)
        with self.io_lock:
            self.hand.set_forces(safe_forces)

        points = []
        started = time.time()
        period = 1.0 / sample_hz
        tactile_period = 1.0 / tactile_hz
        tactile_available = callable(
            getattr(self.hand.t, "read_tactile", None)
        )

        def read_tactile_values():
            return [
                value
                for region in ("index_tip", "index_nail", "index_pad")
                for row in self.hand.t.read_tactile(region)
                for value in row
            ]

        baseline_forces = []
        for _ in range(baseline_samples):
            with self.io_lock:
                baseline_forces.append(self.hand.get_forces()[index])
            time.sleep(period)
        baseline_force = sum(baseline_forces) / len(baseline_forces)
        baseline_tactile = []
        if tactile_available:
            try:
                with self.io_lock:
                    baseline_tactile = read_tactile_values()
            except Exception:
                tactile_available = False
        last_tactile = list(baseline_tactile)
        next_tactile_time = time.time() + tactile_period

        def sample(phase, force_tactile=False):
            nonlocal tactile_available, last_tactile, next_tactile_time
            sample_time = time.time()
            with self.io_lock:
                actual = self.hand.get_angles()[index]
                actual_position = self.hand.t.read_shorts("POS_ACT")[index]
                raw_force = self.hand.get_forces()[index]
                status = self.hand.get_status()[index]
                tactile_peak = ""
                tactile_delta = ""
                tactile_updated = 0
                if (tactile_available
                        and (force_tactile
                             or sample_time >= next_tactile_time)):
                    try:
                        last_tactile = read_tactile_values()
                        tactile_updated = 1
                        next_tactile_time = sample_time + tactile_period
                    except Exception:
                        tactile_available = False
                if last_tactile:
                    tactile_peak = max(last_tactile, default=0)
                    if len(last_tactile) == len(baseline_tactile):
                        tactile_delta = max(
                            abs(value - baseline)
                            for value, baseline
                            in zip(last_tactile, baseline_tactile)
                        )
            point = {
                "timestamp": round(sample_time, 3),
                "elapsed_s": round(sample_time - started, 3),
                "phase": phase,
                "command_angle": int(deepest_command),
                "actual_angle": int(actual),
                "actual_position": int(actual_position),
                "raw_force_g": int(raw_force),
                "net_force_g": round(abs(raw_force - baseline_force), 3),
                "status": int(status),
                "tactile_peak": tactile_peak,
                "tactile_delta": tactile_delta,
                "tactile_updated": tactile_updated,
            }
            points.append(point)
            return point

        command_sent = False
        completed = False
        try:
            # The only calibration movement command.
            self.move("index", int(deepest_command), speed=probe_speed,
                      wait=False)
            command_sent = True
            stable_samples = 0
            previous_angle = initial_angle
            deadline = time.time() + timeout
            stop_reason = "timeout"
            while time.time() < deadline:
                loop_started = time.time()
                point = sample("motion")
                actual = point["actual_angle"]
                if abs(actual - deepest_command) <= 15:
                    stop_reason = "reached zero command"
                    break
                if point["status"] in (3, 5, 6, 7):
                    stop_reason = "firmware force/protection stop"
                    break
                if abs(actual - previous_angle) <= 1:
                    stable_samples += 1
                else:
                    stable_samples = 0
                if (actual < initial_angle - 10 and stable_samples >= 5):
                    stop_reason = "actual angle stopped changing"
                    break
                previous_angle = actual
                time.sleep(max(0.0, period - (time.time() - loop_started)))
            deepest_point = sample("deepest", force_tactile=True)
            deepest_angle = int(deepest_point["actual_angle"])

            if log_path is None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                log_path = Path("logs") / f"drill_calibration_{stamp}.csv"
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fields = ["timestamp", "elapsed_s", "phase", "command_angle",
                      "actual_angle", "actual_position", "raw_force_g",
                      "net_force_g", "status", "tactile_peak",
                      "tactile_delta", "tactile_updated", "contact_angle",
                      "deepest_angle"]
            with log_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for point in points:
                    row = dict(point)
                    row["contact_angle"] = ""
                    row["deepest_angle"] = deepest_angle
                    writer.writerow(row)

            probe = {
                "deepest_angle": deepest_angle,
                "release_angle": int(start_angle),
                "baseline_force_g": round(baseline_force, 3),
                "max_force_g": max_force_g,
                "stop_reason": stop_reason,
                "effective_sample_hz": round(
                    (len(points) - 1)
                    / max(0.001,
                          points[-1]["elapsed_s"] - points[0]["elapsed_s"]),
                    1,
                ),
                "log_path": str(log_path),
                "points": points,
            }
            self.drill_probe = probe
            self.drill_calibration = None
            completed = True
            return probe
        finally:
            try:
                if command_sent and not completed:
                    self.move("index", int(start_angle), speed=probe_speed,
                              wait=True)
            finally:
                with self.io_lock:
                    self.hand.set_speeds(original_speeds)
                    self.hand.set_forces(original_forces)

    def set_drill_calibration(self, contact_angle, deepest_angle=None):
        """Apply contact/deepest angles chosen from the one-motion graph."""
        probe = self.drill_probe
        if deepest_angle is None:
            if probe is None:
                raise RuntimeError(
                    "no drill probe available; provide deepest_angle or run "
                    "drillcal first"
                )
            deepest_angle = probe["deepest_angle"]
        contact = int(contact_angle)
        deepest = int(deepest_angle)
        if not (0 <= deepest < contact <= 1000):
            raise ValueError(
                "require 0 <= deepest_angle < contact_angle <= 1000"
            )
        calibration = {
            "contact_angle": contact,
            "deepest_angle": deepest,
            "span": contact - deepest,
            "release_angle": (probe["release_angle"] if probe else 1000),
            "log_path": probe["log_path"] if probe else None,
        }
        self.drill_calibration = calibration
        if probe and probe.get("log_path"):
            path = Path(probe["log_path"])
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                rows = list(reader)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row["contact_angle"] = contact
                    row["deepest_angle"] = deepest
                    writer.writerow(row)
            temporary.replace(path)
        return calibration

    def drill_level_angle(self, level_percent):
        """Convert a 0-100% trigger level to a calibrated index angle."""
        calibration = getattr(self, "drill_calibration", None)
        if calibration is None:
            raise RuntimeError("run drillcal before setting a drill level")
        level = float(level_percent)
        if not (0 <= level <= 100):
            raise ValueError("drill level must be between 0 and 100 percent")
        contact = calibration["contact_angle"]
        span = calibration["span"]
        return round(contact - (level / 100.0) * span)

    def set_drill_level(self, level_percent, speed=150, wait=True):
        """Press the drill trigger to a calibrated percentage using index."""
        target = self.drill_level_angle(level_percent)
        self.move("index", target, speed=speed, wait=wait)
        return target

    def release_drill_trigger(self, speed=200, wait=True):
        """Move index clear of the trigger; safe even before calibration."""
        calibration = getattr(self, "drill_calibration", None)
        target = calibration["release_angle"] if calibration else 1000
        self.move("index", target, speed=speed, wait=wait)
        return target

    def wave_test(self, angle_down=200, delay=0.6):
        """Sequentially bend and release each finger — quick hardware check."""
        for name in ["little", "ring", "middle", "index", "thumb"]:
            print(f"  bending {name} ...")
            self.move(name, angle_down, wait=True)
            time.sleep(delay)
            self.move(name, 1000, wait=True)


# ===========================================================================
# Plotting
# ===========================================================================
def plot_tactile_snapshot(tactile, output, title="Tactile snapshot"):
    """Save one full-hand tactile snapshot as spatial heatmaps."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot tactile marks. "
                         "Install it: pip install matplotlib") from exc

    regions = [(tactile_label(name), tactile[name])
               for name in TACTILE if name in tactile]
    if not regions:
        raise ValueError("tactile snapshot contains no regions")
    global_max = max(
        (value for _, matrix in regions for row in matrix for value in row),
        default=1,
    )
    plot_cols = 5
    plot_rows = (len(regions) + plot_cols - 1) // plot_cols
    fig = Figure(figsize=(3.0 * plot_cols, 2.7 * plot_rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(plot_rows, plot_cols)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    image = None
    for ax, (region, matrix) in zip(axes, regions):
        image = ax.imshow(matrix, cmap="inferno", vmin=0,
                          vmax=max(1, global_max), interpolation="nearest",
                          aspect="equal", origin="upper")
        ax.set_title(region.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(regions):]:
        ax.set_visible(False)
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
    fig.colorbar(image, cax=colorbar_ax, label="raw tactile value")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04,
                        top=0.90, wspace=0.28, hspace=0.35)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    fig.clear()
    return str(output)


def plot_drill_calibration(path=None, show=True):
    """Plot a raw one-motion drill probe or its selected calibration map."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot calibration. "
                         "Install it: pip install matplotlib") from exc

    if path is None:
        logs = sorted(Path("logs").glob("drill_calibration_*.csv"))
        if not logs:
            raise FileNotFoundError("no logs/drill_calibration_*.csv found")
        path = logs[-1]
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no calibration rows")

    deepest = float(rows[0]["deepest_angle"])
    actual = [float(row["actual_angle"]) for row in rows]
    force = [float(row["net_force_g"]) for row in rows]
    elapsed = [float(row["elapsed_s"]) for row in rows]
    tactile = [float(row["tactile_delta"])
               if row.get("tactile_delta", "") != "" else 0.0
               for row in rows]
    contact_text = rows[0].get("contact_angle", "")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    if contact_text == "":
        # Raw one-motion view: choose first contact where force/tactile begins
        # its sustained rise, then enter that angle with drillsetcal.
        axes[0].plot(elapsed, actual, color="tab:blue", linewidth=1.6)
        axes[0].axhline(deepest, color="tab:red", linestyle="--",
                        label=f"deepest actual angle: {deepest:.0f}")
        axes[0].set_title("One continuous index motion: released to command 0")
        axes[0].set_xlabel("elapsed time (s)")
        axes[0].set_ylabel("actual index angle")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        force_line = axes[1].plot(actual, force, color="tab:orange",
                                  label="force change (g)")[0]
        tactile_axis = axes[1].twinx()
        tactile_line = tactile_axis.plot(
            actual, tactile, color="tab:purple", label="tactile change"
        )[0]
        axes[1].axvline(deepest, color="tab:red", linestyle="--")
        axes[1].invert_xaxis()
        axes[1].set_title(
            "Select first-contact angle at the sustained sensor rise"
        )
        axes[1].set_xlabel("actual index angle (motion proceeds left to right)")
        axes[1].set_ylabel("force change from baseline (g)")
        tactile_axis.set_ylabel("maximum tactile-cell change")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend([force_line, tactile_line],
                       ["force change (g)", "tactile change"], loc="best")
    else:
        contact = float(contact_text)
        span = contact - deepest
        if span <= 0:
            raise ValueError(f"{path} contains an invalid calibration span")
        levels = list(range(101))
        targets = [contact - level / 100.0 * span for level in levels]
        axes[0].plot(levels, targets, color="tab:blue", linewidth=2)
        marker_levels = [0, 25, 50, 75, 100]
        marker_targets = [contact - level / 100.0 * span
                          for level in marker_levels]
        axes[0].scatter(marker_levels, marker_targets,
                        color="tab:blue", zorder=3)
        for level, target in zip(marker_levels, marker_targets):
            axes[0].annotate(f"{target:.0f}", (level, target),
                             textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=8)
        axes[0].set_title("Calibrated drill-trigger command map")
        axes[0].set_xlabel("requested trigger level (%)")
        axes[0].set_ylabel("index target angle\n(1000=open, 0=closed)")
        axes[0].grid(True, alpha=0.3)

        depth = [max(0.0, contact - angle) for angle in actual]
        scatter = axes[1].scatter(depth, force, c=elapsed,
                                  cmap="viridis", s=25)
        axes[1].plot(depth, force, color="0.65", linewidth=0.8, zorder=0)
        axes[1].axvline(0, color="tab:green", linestyle="--",
                        label=f"first contact: {contact:.0f}")
        axes[1].axvline(span, color="tab:red", linestyle="--",
                        label=f"full press: {deepest:.0f}")
        axes[1].set_title("Measured force across calibrated trigger travel")
        axes[1].set_xlabel("button press depth (angle units from contact)")
        axes[1].set_ylabel("force change from baseline (g)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.colorbar(scatter, ax=axes[1], label="probe elapsed time (s)")
    fig.suptitle(path.name)
    fig.tight_layout()

    output = path.with_name(f"{path.stem}_plot.png")
    fig.savefig(output, dpi=140)
    print(f"  saved drill calibration plot -> {output}")
    if show:
        plt.show()
    plt.close(fig)
    return str(output)


def _plot_tactile_heatmaps(path, idx, cols, plt):
    """Save peak-over-time spatial heatmaps for every logged tactile region."""
    matrices = {}
    global_max = 0.0
    for region, (_, _, rows, columns) in TACTILE.items():
        matrix = []
        found = False
        for r in range(rows):
            matrix_row = []
            for c in range(columns):
                key = f"tactile_{region}_r{r}_c{c}"
                if key not in idx:
                    matrix_row.append(0.0)
                    continue
                values = [float(v) for v in cols[idx[key]] if v != ""]
                peak = max(values, default=0.0)
                matrix_row.append(peak)
                global_max = max(global_max, peak)
                found = True
            matrix.append(matrix_row)
        if found:
            matrices[region] = matrix

    if not matrices:
        return None

    plot_cols = 5
    plot_rows = (len(matrices) + plot_cols - 1) // plot_cols
    fig, axes = plt.subplots(plot_rows, plot_cols,
                             figsize=(3.0 * plot_cols, 2.7 * plot_rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    image = None
    for ax, (region, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap="inferno", vmin=0,
                          vmax=max(1.0, global_max), interpolation="nearest",
                          aspect="equal", origin="upper")
        ax.set_title(region.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(matrices):]:
        ax.set_visible(False)
    fig.suptitle(f"Peak tactile response by sensor cell — {path.name}")
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
    fig.colorbar(image, cax=colorbar_ax, label="raw tactile value")
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04,
                        top=0.90, wspace=0.28, hspace=0.35)
    out = path.with_name(f"{path.stem}_tactile.png")
    fig.savefig(out, dpi=140)
    print(f"  saved tactile heatmaps -> {out}")
    return fig, out


def _h5_read_parameters(file, root=""):
    """Read per-finger motor datasets back into ``{channel: (sample, DOF)}``.

    The file stores one 1D dataset per finger, which is what makes it readable;
    the plotting code wants them stacked in DOF order, so rebuild that here.
    """
    _, np = _require_h5py()
    channels = {}
    for index in sorted(DOF_LAYOUT):
        group = file.get(dof_path(index, root))
        if group is None:
            continue
        for name, dataset in group.items():
            if isinstance(dataset, h5_dataset_types()):
                channels.setdefault(name, {})[index] = dataset[:]
    parameters = {}
    for name, per_dof in channels.items():
        if len(per_dof) != len(DOF_LAYOUT):
            continue
        parameters[name] = np.stack([per_dof[i] for i in sorted(per_dof)],
                                    axis=1)
    return parameters


def h5_dataset_types():
    """h5py.Dataset, imported lazily so the module still loads without h5py."""
    h5py, _ = _require_h5py()
    return (h5py.Dataset,)


def _plot_h5_tactile_heatmaps(path, tactile, plt):
    """Plot per-cell maxima from grouped HDF5 tactile arrays."""
    matrices = {
        tactile_label(region): values.max(axis=0)
        for region, values in tactile.items() if len(values)
    }
    if not matrices:
        return None
    global_max = max(float(matrix.max()) for matrix in matrices.values())
    plot_cols = 5
    plot_rows = (len(matrices) + plot_cols - 1) // plot_cols
    fig, axes = plt.subplots(plot_rows, plot_cols,
                             figsize=(3.0 * plot_cols, 2.7 * plot_rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    image = None
    for ax, (region, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap="inferno", vmin=0,
                          vmax=max(1.0, global_max), interpolation="nearest",
                          aspect="equal", origin="upper")
        ax.set_title(region.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(matrices):]:
        ax.set_visible(False)
    fig.suptitle(f"Peak tactile response by sensor cell — {path.name}")
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
    fig.colorbar(image, cax=colorbar_ax, label="raw tactile value")
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04,
                        top=0.90, wspace=0.28, hspace=0.35)
    out = path.with_name(f"{path.stem}_tactile.png")
    fig.savefig(out, dpi=140)
    print(f"  saved tactile heatmaps -> {out}")
    return fig, out


def _available_hand_logs():
    """Return legacy and session-organized hand logs in time order."""
    paths = list(Path("logs").glob("hand_*.csv"))
    paths += list(Path("data_logs").glob("session_*/hand.csv"))
    paths += list(Path("data_logs").glob("session_*/hand.h5"))
    paths += list(Path("logs").glob("hand_*.h5"))
    return sorted(paths, key=lambda path: path.stat().st_mtime)


def export_marked_data(session=None, output=None):
    """Export exact HDF5 marks (with legacy CSV session support).

    ``session`` may be a session directory or its ``hand.h5``. The default
    output is ``marked_data.h5`` and retains grouped parameter/tactile arrays.
    """
    if session is None:
        sessions = [path for path in Path("data_logs").glob("session_*")
                    if path.is_dir()]
        if not sessions:
            raise FileNotFoundError("no data_logs/session_* directory found")
        session_dir = max(sessions, key=lambda path: path.stat().st_mtime)
    else:
        session_path = Path(session)
        session_dir = (session_path.parent
                       if session_path.is_file() else session_path)
    source_h5 = (session_path if session is not None
                 and session_path.is_file()
                 and session_path.suffix.lower() in (".h5", ".hdf5")
                 else session_dir / "hand.h5")
    if source_h5.exists():
        h5py, _ = _require_h5py()
        output = (Path(output) if output
                  else session_dir / "marked_data.h5")
        if output.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("marked HDF5 output must use .h5 or .hdf5")
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source_h5, "r") as source:
            if "marks" not in source or not len(source["marks/id"]):
                raise ValueError(f"{source_h5} contains no marked points")
            with h5py.File(output, "w") as target:
                target.attrs["format"] = "RH56_MARKED_DATA"
                target.attrs["format_version"] = 2
                target.attrs["source_log"] = str(source_h5)
                _write_time_attrs(target, time.time(), "exported")
                for name in ("created_iso", "created_date", "created_time",
                             "timezone"):
                    if name in source.attrs:
                        target.attrs[f"source_{name}"] = source.attrs[name]
                source.copy("marks", target)
        print(f"  exported HDF5 marks -> {output}")
        return str(output)
    marks_path = session_dir / "marks.csv"
    if not marks_path.exists():
        raise FileNotFoundError(f"no marks.csv found in {session_dir}")

    with marks_path.open(newline="", encoding="utf-8") as file:
        marks = list(csv.DictReader(file))
    if not marks:
        raise ValueError(f"{marks_path} contains no marked points")

    exported_rows = []
    fieldnames = None
    for mark in marks:
        snapshot_path = Path(mark["snapshot_csv"])
        if not snapshot_path.exists():
            snapshot_path = session_dir / "marks" / snapshot_path.name
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"snapshot for mark {mark['mark_id']} not found: "
                f"{snapshot_path}"
            )
        with snapshot_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            snapshot_rows = list(reader)
            if fieldnames is None:
                fieldnames = reader.fieldnames
        if len(snapshot_rows) != 1:
            raise ValueError(
                f"expected one row in marked snapshot {snapshot_path}"
            )
        exported_rows.append(snapshot_rows[0])

    output = Path(output) if output else session_dir / "marked_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported_rows)
    print(f"  exported {len(exported_rows)} marked points -> {output}")
    return str(output)


def _export_h5_tactile_video(path, output, fps, plt, animation):
    """Render grouped HDF5 tactile frames without flattening their pads."""
    h5py, _ = _require_h5py()
    with h5py.File(path, "r") as file:
        elapsed = file["time/tactile_elapsed_s"][:]
        tactile = {tactile_label(region): file[tactile_path(region)][:]
                   for region in TACTILE if tactile_path(region) in file}
    tactile = {name: frames for name, frames in tactile.items() if len(frames)}
    if not len(elapsed) or not tactile:
        raise ValueError(f"{path} contains no tactile frames")
    ffmpeg_available = animation.writers.is_available("ffmpeg")
    if output:
        output = Path(output)
    else:
        suffix = ".mp4" if ffmpeg_available else ".gif"
        output = path.with_name(f"{path.stem}_tactile{suffix}")
    if output.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("tactile video output must use .mp4 or .gif")
    positive = sorted(float(value) for frames in tactile.values()
                      for value in frames.reshape(-1) if value > 0)
    color_max = (max(1.0, positive[int(.995 * (len(positive) - 1))])
                 if positive else 1.0)
    plot_cols = 5
    plot_rows = (len(tactile) + plot_cols - 1) // plot_cols
    fig, axes = plt.subplots(plot_rows, plot_cols,
                             figsize=(3.0 * plot_cols, 2.7 * plot_rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    images = []
    for ax, (region, frames) in zip(axes, tactile.items()):
        image = ax.imshow(frames[0], cmap="inferno", vmin=0, vmax=color_max,
                          interpolation="nearest", aspect="equal",
                          origin="upper")
        images.append((image, frames))
        ax.set_title(region.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(tactile):]:
        ax.set_visible(False)
    time_text = fig.text(0.5, 0.955, "", ha="center", va="top", fontsize=11)
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
    fig.colorbar(images[0][0], cax=colorbar_ax, label="raw tactile value")
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04,
                        top=0.90, wspace=0.28, hspace=0.35)

    def update(frame_number):
        for image, frames in images:
            image.set_data(frames[frame_number])
        time_text.set_text(
            f"RH56 tactile map — {path.name} — "
            f"t = {elapsed[frame_number]:.3f} s"
        )
        return [*(image for image, _ in images), time_text]

    if output.suffix.lower() == ".mp4" and not ffmpeg_available:
        plt.close(fig)
        raise RuntimeError("FFmpeg was not found; choose a .gif output.")
    movie = animation.FuncAnimation(fig, update, frames=len(elapsed),
                                    interval=1000.0 / fps, blit=False)
    writer = (animation.FFMpegWriter(fps=fps, codec="h264", bitrate=3000)
              if output.suffix.lower() == ".mp4"
              else animation.PillowWriter(fps=fps))
    movie.save(output, writer=writer,
               dpi=110 if output.suffix.lower() == ".mp4" else 80)
    plt.close(fig)
    print(f"  saved tactile video -> {output} ({len(elapsed)} frames)")
    return str(output)


def export_tactile_video(path=None, output=None, fps=5.0):
    """Export logged tactile sensor frames as an MP4 or animated GIF.

    HDF5 tactile frames use their independent tactile timeline. For legacy
    CSV, only rows with ``tactile_updated == 1`` are rendered, because rows
    between them carry a repeated tactile frame for time alignment with the
    faster motor log.  All regions share one color scale, making contact
    intensity comparable across the hand.

    ``path`` defaults to the newest ``logs/hand_*.csv``. If ``output`` is
    omitted, MP4 is used when FFmpeg is available and animated GIF otherwise.
    Set ``FFMPEG_PATH`` if FFmpeg is installed but is not on PATH.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        if ffmpeg_path:
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError as exc:
        raise SystemExit("matplotlib is required to export tactile video. "
                         "Install it: pip install matplotlib") from exc

    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if path is None:
        logs = _available_hand_logs()
        if not logs:
            raise FileNotFoundError("no HDF5 or legacy CSV hand log found")
        path = logs[-1]
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return _export_h5_tactile_video(
            path, output, fps, plt, animation
        )
    ffmpeg_available = animation.writers.is_available("ffmpeg")
    if output:
        output = Path(output)
    else:
        suffix = ".mp4" if ffmpeg_available else ".gif"
        output = path.with_name(f"{path.stem}_tactile{suffix}")
    if output.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("tactile video output must use .mp4 or .gif")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path} has no data rows")
    idx = {name: column for column, name in enumerate(header)}
    if "elapsed_s" not in idx:
        raise ValueError(f"{path} has no elapsed_s column")

    tactile_keys = []
    for region, (_, _, region_rows, region_cols) in TACTILE.items():
        keys = [[f"tactile_{region}_r{r}_c{c}"
                 for c in range(region_cols)]
                for r in range(region_rows)]
        if all(key in idx for matrix_row in keys for key in matrix_row):
            tactile_keys.append((region, keys))
    if not tactile_keys:
        raise ValueError(f"{path} contains no tactile-cell columns")

    updated_column = idx.get("tactile_updated")
    frames = [row for row in rows
              if updated_column is None or row[updated_column] == "1"]
    frames = [row for row in frames
              if any(row[idx[key]] != ""
                     for _, keys in tactile_keys
                     for matrix_row in keys for key in matrix_row)]
    if not frames:
        raise ValueError(f"{path} contains no populated tactile frames")

    # A high percentile keeps a single noisy sensor from washing out every
    # other cell, while retaining one common scale across regions and time.
    positive_values = sorted(
        float(row[idx[key]])
        for row in frames
        for _, keys in tactile_keys
        for matrix_row in keys for key in matrix_row
        if row[idx[key]] not in ("", "0") and float(row[idx[key]]) > 0
    )
    if positive_values:
        scale_index = int(0.995 * (len(positive_values) - 1))
        color_max = max(1.0, positive_values[scale_index])
    else:
        color_max = 1.0

    plot_cols = 5
    plot_rows = (len(tactile_keys) + plot_cols - 1) // plot_cols
    fig, axes = plt.subplots(plot_rows, plot_cols,
                             figsize=(3.0 * plot_cols, 2.7 * plot_rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    images = []
    for ax, (region, keys) in zip(axes, tactile_keys):
        empty = [[0.0 for _ in matrix_row] for matrix_row in keys]
        image = ax.imshow(empty, cmap="inferno", vmin=0, vmax=color_max,
                          interpolation="nearest", aspect="equal",
                          origin="upper")
        images.append(image)
        ax.set_title(region.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(tactile_keys):]:
        ax.set_visible(False)
    time_text = fig.text(0.5, 0.955, "", ha="center", va="top", fontsize=11)
    colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
    fig.colorbar(images[0], cax=colorbar_ax, label="raw tactile value")
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04,
                        top=0.90, wspace=0.28, hspace=0.35)

    def update(frame_number):
        row = frames[frame_number]
        for image, (_, keys) in zip(images, tactile_keys):
            matrix = [[float(row[idx[key]]) if row[idx[key]] != "" else 0.0
                       for key in matrix_row] for matrix_row in keys]
            image.set_data(matrix)
        elapsed = float(row[idx["elapsed_s"]])
        time_text.set_text(
            f"RH56 tactile map — {path.name} — t = {elapsed:.3f} s"
        )
        return [*images, time_text]

    if output.suffix.lower() == ".mp4" and not ffmpeg_available:
        plt.close(fig)
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg or set FFMPEG_PATH to its "
            "executable, or choose a .gif output."
        )
    movie = animation.FuncAnimation(fig, update, frames=len(frames),
                                    interval=1000.0 / fps, blit=False)
    if output.suffix.lower() == ".mp4":
        writer = animation.FFMpegWriter(
            fps=fps, codec="h264", bitrate=3000,
            metadata={"title": f"RH56 tactile data: {path.name}"},
        )
        dpi = 110
    else:
        writer = animation.PillowWriter(fps=fps)
        dpi = 80
    movie.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  saved tactile video -> {output} ({len(frames)} frames at {fps:g} fps)")
    return str(output)


def _plot_log_h5(path, show, plt):
    """Plot the native grouped HDF5 hand-log layout."""
    h5py, _ = _require_h5py()
    with h5py.File(path, "r") as file:
        if file.attrs.get("format", "") != "RH56_HAND_LOG":
            raise ValueError(f"{path} is not an RH56 hand log")
        t = file["time/elapsed_s"][:]
        parameters = _h5_read_parameters(file)
        tactile_time = file["time/tactile_elapsed_s"][:]
        tactile = {region: file[tactile_path(region)][:]
                   for region in TACTILE if tactile_path(region) in file}
        recorded_iso = file.attrs.get("created_iso", "")
        marks = []
        if "marks" in file:
            labels = file["marks/label"].asstr()[:]
            marks = list(zip(file["marks/elapsed_s"][:], labels))
    if not len(t):
        raise ValueError(f"{path} has no data samples")

    all_groups = [("angle", "angle (0-1000)", False),
                  ("pos", "position (0-2000)", False),
                  ("force_g", "net force (g)", False),
                  ("speed", "speed (0-1000)", False),
                  ("current_mA", "current (mA)", False),
                  ("temp_C", "temp (°C)", False),
                  ("status", "status (0-7)", True)]
    groups = [group for group in all_groups if group[0] in parameters]
    has_tactile = bool(len(tactile_time) and
                       any(len(values) for values in tactile.values()))
    subplot_count = len(groups) + (1 if has_tactile else 0)
    fig, axes = plt.subplots(subplot_count, 1, sharex=True,
                             figsize=(11, 1.9 * subplot_count + 1))
    if subplot_count == 1:
        axes = [axes]
    for ax, (name, ylabel, step) in zip(axes, groups):
        values = parameters[name].astype(float)
        for index, finger in enumerate(FINGER_NAMES):
            y = values[:, index]
            if name == "force_g":
                y = y - statistics.median(y[:20])
            ax.plot(t, y, label=finger, linewidth=1.2,
                    drawstyle="steps-post" if step else "default")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    if has_tactile:
        tactile_ax = axes[-1]
        by_finger = {}
        for region, frames in tactile.items():
            if len(frames):
                peak = frames.reshape(len(frames), -1).max(axis=1)
                part = TACTILE_LAYOUT[region][0]
                by_finger.setdefault(part, []).append(peak)
        for part, peaks in by_finger.items():
            combined = peaks[0]
            for peak in peaks[1:]:
                combined = combined.copy()
                combined[peak > combined] = peak[peak > combined]
            tactile_ax.plot(tactile_time, combined, label=part, linewidth=1.1)
        tactile_ax.set_ylabel("tactile peak")
        tactile_ax.grid(True, alpha=0.3)
        tactile_ax.legend(loc="upper right", ncol=6, fontsize=8)
    for elapsed, label in marks:
        for ax in axes:
            ax.axvline(elapsed, color="magenta", linestyle="--",
                       linewidth=1.0, alpha=0.75)
        axes[0].annotate(label, (elapsed, 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(3, -3), textcoords="offset points",
                         rotation=90, va="top", ha="left", fontsize=8,
                         color="magenta")
    title = f"RH56 hand parameters — {path.name}"
    if recorded_iso:
        title += f"  ({recorded_iso})"
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", ncol=6, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"  saved plot -> {out}")
    tactile_plot = _plot_h5_tactile_heatmaps(path, tactile, plt)
    if show:
        plt.show()
    plt.close(fig)
    if tactile_plot:
        plt.close(tactile_plot[0])
    return str(out)


def plot_log(path=None, show=True):
    """Plot an HDF5 hand log (and legacy CSV logs).

    path: HDF5/CSV log; defaults to the most recent hand log.
    Saves the time-series PNG plus a ``*_tactile.png`` peak heatmap when the
    The log's tactile data also produces a peak heatmap PNG.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")            # headless: file only, no window
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot. "
                         "Install it:  pip install matplotlib") from exc

    if path is None:
        logs = _available_hand_logs()
        if not logs:
            raise FileNotFoundError("no hand log found — log something first")
        path = logs[-1]
    path = Path(path)

    if path.suffix.lower() in (".h5", ".hdf5"):
        return _plot_log_h5(path, show, plt)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    if not rows:
        raise ValueError(f"{path} has no data rows")

    idx = {name: k for k, name in enumerate(header)}
    cols = list(zip(*rows))
    t = [float(v) for v in cols[idx["elapsed_s"]]]

    # All 8 logged parameters. status/error are step plots (state code /
    # bitfield) rather than smooth lines.
    all_groups = [("angle", "angle (0-1000)", False),
                  ("pos", "position (0-2000)", False),
                  ("force_g", "net force (g)", False),
                  ("speed", "speed (0-1000)", False),
                  ("current_mA", "current (mA)", False),
                  ("temp_C", "temp (°C)", False),
                  ("status", "status (0-7)", True)]
    # only plot a parameter if this CSV actually has its columns
    groups = [g for g in all_groups
              if any(f"{g[0]}_{n}" in idx for n in FINGER_NAMES)]

    tactile_columns = {}
    for region, (_, _, rows_count, cols_count) in TACTILE.items():
        part = region.split("_", 1)[0]
        for r in range(rows_count):
            for c in range(cols_count):
                key = f"tactile_{region}_r{r}_c{c}"
                if key in idx and any(v != "" for v in cols[idx[key]]):
                    tactile_columns.setdefault(part, []).append(idx[key])
    has_tactile = bool(tactile_columns)

    subplot_count = len(groups) + (1 if has_tactile else 0)
    fig, axes = plt.subplots(subplot_count, 1, sharex=True,
                             figsize=(11, 1.9 * subplot_count + 1))
    if subplot_count == 1:
        axes = [axes]
    for ax, (prefix, ylabel, step) in zip(axes, groups):
        for n in FINGER_NAMES:
            key = f"{prefix}_{n}"
            if key in idx:
                y = [float(v) for v in cols[idx[key]]]
                if prefix == "force_g":
                    baseline = statistics.median(y[:20])
                    y = [value - baseline for value in y]
                ax.plot(t, y, label=n, linewidth=1.2,
                        drawstyle="steps-post" if step else "default")
        ax.set_ylabel(ylabel)
        if prefix == "force_g":
            ax.set_title(
                "Net force (raw FORCE_ACT minus first 20-sample median)"
            )
        ax.grid(True, alpha=0.3)
    if has_tactile:
        tactile_ax = axes[-1]
        for part, column_indices in tactile_columns.items():
            peak_over_time = []
            for sample in range(len(rows)):
                values = [float(cols[k][sample])
                          for k in column_indices if cols[k][sample] != ""]
                peak_over_time.append(max(values, default=0.0))
            tactile_ax.plot(t, peak_over_time, label=part, linewidth=1.1)
        tactile_ax.set_ylabel("tactile peak")
        tactile_ax.grid(True, alpha=0.3)
        tactile_ax.legend(loc="upper right", ncol=6, fontsize=8)
    if "mark_id" in idx:
        mark_label_column = idx.get("mark_label")
        for sample, mark_ids in enumerate(cols[idx["mark_id"]]):
            if not mark_ids:
                continue
            label = (cols[mark_label_column][sample]
                     if mark_label_column is not None else mark_ids)
            for ax in axes:
                ax.axvline(t[sample], color="magenta", linestyle="--",
                           linewidth=1.0, alpha=0.75)
            axes[0].annotate(
                label, (t[sample], 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", rotation=90,
                va="top", ha="left", fontsize=8, color="magenta",
            )
    axes[0].set_title(f"RH56 hand parameters — {path.name}")
    axes[0].legend(loc="upper right", ncol=6, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()

    out = path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"  saved plot -> {out}")
    tactile_plot = _plot_tactile_heatmaps(path, idx, cols, plt)
    if show:
        plt.show()
    plt.close(fig)
    if tactile_plot:
        plt.close(tactile_plot[0])
    return str(out)


# ===========================================================================
# Interactive CLI
# ===========================================================================
SHORTCUT_FILE = Path(__file__).resolve().with_name("finger_shortcuts.json")
RESERVED_SHORTCUT_NAMES = {
    "set", "unset", "shortcuts", "quit", "exit", "q", "help", "status",
    "angles", "forces", "speeds", "log", "logseq", "plot",
    "mark", "plottactile", "exportmarks", "tactilevideo", "speed",
    "speedall", "force", "forceall", "openall",
    "closeall", "wave", "group", "seq", "all", "bend", "open", "state",
    "drillcal", "drillsetcal", "drill", "drillrelease", "drillplot",
    *FINGERS.keys(),
}


def load_shortcuts(path=SHORTCUT_FILE):
    """Load persistent CLI shortcuts as ``name -> command token list``."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read shortcut file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"shortcut file {path} must contain a JSON object")
    shortcuts = {}
    for name, command in data.items():
        if (isinstance(name, str) and isinstance(command, list)
                and command and all(isinstance(token, str)
                                    for token in command)):
            shortcuts[name.lower()] = command
    return shortcuts


def save_shortcuts(shortcuts, path=SHORTCUT_FILE):
    """Atomically save CLI shortcuts so they survive program restarts."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(shortcuts, indent=2, ensure_ascii=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_shortcut_name(name):
    """Return a normalized shortcut name or raise a useful error."""
    name = name.strip().lower()
    if not name:
        raise ValueError("shortcut name cannot be empty")
    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError("shortcut name must start with a letter or underscore")
    if not all(char.isalnum() or char in "_-" for char in name):
        raise ValueError(
            "shortcut name may contain only letters, numbers, '_' and '-'"
        )
    if name in RESERVED_SHORTCUT_NAMES:
        raise ValueError(f"'{name}' is already a built-in command or finger")
    return name


def expand_shortcut(tokens, shortcuts):
    """Expand a shortcut, supporting chained shortcuts while rejecting loops."""
    expanded = list(tokens)
    seen = set()
    while expanded and expanded[0].lower() in shortcuts:
        name = expanded[0].lower()
        if len(expanded) != 1:
            raise ValueError(f"shortcut '{name}' does not accept arguments")
        if name in seen:
            raise ValueError(f"shortcut loop detected at '{name}'")
        seen.add(name)
        expanded = list(shortcuts[name])
    return expanded


HELP = """
Commands (finger = little/ring/middle/index/thumb/thumb_rot or 0-5):
  set <name> <command...>     save a persistent shortcut
                              e.g. set drill_grasp group middle 0 ring 0
  unset <name>               delete a saved shortcut
  shortcuts                  list all saved shortcuts
  drillcal [max_force_g]     one motion to 0; speed=1000, force=3000 g default
                              WARNING: this can start and hold the drill
  drillsetcal <contact> [deepest]
                              set endpoints chosen from the drillcal graph
  drill <level_percent>      set calibrated trigger level from 0 to 100
  drillrelease               move index completely clear of the trigger
  drillplot [path]           plot newest/specified drill calibration CSV
  <finger> <angle>            e.g.  index 300      thumb_rot 800
  <finger> <angle> <force>    e.g.  index 300 150  (sets force first)
  speed <finger> <value>      set one finger's speed (0-1000), no move
  speedall <value>            set ALL fingers' speed (0-1000), no move
  force <finger> <value>      set one finger's force limit (grams), no move
  forceall <value>            set ALL fingers' force limit (grams), no move
  bend <finger> | open <finger>
  group <finger> <angle> [<finger> <angle> ...]
                              move several fingers together (they hold until
                              your next command)  e.g.  group index 300 middle 300
  seq <finger> <angle> [delay <milliseconds> | <finger> <angle> ...]
                              move fingers one-by-one with optional pauses
                              e.g. seq index 1000 delay 1000 index 500
                                   delay 1000 index 0 delay 1000
  state <finger>              show one finger's full state
  all <angle>                 move all six DOF to same angle
  openall | closeall
  wave                        finger-by-finger test sequence
  status                      all fingers' status text
  angles / forces / speeds    read current values
  log start [path]            begin HDF5 recording; default data_logs/session_*/
                              (angle,pos,force,speed,current,temp,status,error)
  mark [label]                store exact HDF5 POI + tactile PNG snapshot
                              e.g. mark trigger_first_contact
  plottactile [path]          read tactile once, save that frame as HDF5 + PNG
                              e.g. plottactile
                                   plottactile experiment/force_50.h5
  exportmarks [session] [out] export exact marks to marked_data.h5
  log stop                    stop recording
  logseq <finger> <angle> ... run a sequence with logging on for its duration
  plot [path]                 plot an HDF5/legacy CSV log -> PNG + window
  tactilevideo [path] [fps]  export tactile frames to MP4 (or GIF fallback)
                              e.g. tactilevideo data_logs/session_*/hand.h5 5
  help | quit
"""


def cli(fc: FingerController):
    try:
        shortcuts = load_shortcuts()
    except ValueError as exc:
        print(f"  warning: {exc}")
        shortcuts = {}
    print(HELP)
    while True:
        try:
            line = input("hand> ").strip()   # keep case for file paths
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        try:
            tok = line.split()
            cmd = tok[0].lower()

            if cmd == "set":
                if len(tok) < 3:
                    print("  usage: set <name> <command...>")
                else:
                    name = validate_shortcut_name(tok[1])
                    command = tok[2:]
                    if command[0].lower() in ("set", "unset", "shortcuts"):
                        raise ValueError(
                            "shortcuts cannot run shortcut-management commands"
                        )
                    shortcuts[name] = command
                    save_shortcuts(shortcuts)
                    print(f"  saved {name} -> {' '.join(command)}")
                continue
            if cmd == "unset":
                if len(tok) != 2:
                    print("  usage: unset <name>")
                else:
                    name = tok[1].lower()
                    if name not in shortcuts:
                        print(f"  shortcut '{name}' does not exist")
                    else:
                        del shortcuts[name]
                        save_shortcuts(shortcuts)
                        print(f"  removed shortcut '{name}'")
                continue
            if cmd == "shortcuts":
                if shortcuts:
                    for name, command in sorted(shortcuts.items()):
                        print(f"  {name} -> {' '.join(command)}")
                else:
                    print("  no shortcuts saved")
                continue

            expanded = expand_shortcut(tok, shortcuts)
            if expanded != tok:
                print(f"  {tok[0]} -> {' '.join(expanded)}")
                tok = expanded
                cmd = tok[0].lower()

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "drillcal":
                if len(tok) > 2:
                    print("  usage: drillcal [max_force_g]")
                else:
                    max_force = int(tok[1]) if len(tok) == 2 else 3000
                    print("  WARNING: this records one continuous index "
                          "motion to angle 0, may start the drill, and holds "
                          "the deepest position afterward.")
                    print(f"  calibration settings: index speed=1000, "
                          f"force limit={max_force} g, fast sensors=100 Hz, "
                          "tactile=5 Hz")
                    answer = input(
                        "  Secure/remove the bit. Continue? [y/N] "
                    ).strip().lower()
                    if answer not in ("y", "yes"):
                        print("  calibration cancelled")
                    else:
                        result = fc.calibrate_drill_trigger(
                            max_force_g=max_force
                        )
                        plot_path = plot_drill_calibration(
                            result["log_path"], show=False
                        )
                        print(
                            f"  probe complete: deepest actual angle="
                            f"{result['deepest_angle']}"
                        )
                        print(f"  stop reason: {result['stop_reason']}")
                        print(f"  measured fast-sensor rate: "
                              f"{result['effective_sample_hz']} Hz")
                        print("  inspect the graph, then run: "
                              "drillsetcal <contact_angle>")
                        print("  holding deepest position; use drillrelease "
                              "when needed")
                        print(f"  calibration log -> {result['log_path']}")
                        print(f"  calibration plot -> {plot_path}")
            elif cmd == "drillsetcal":
                if len(tok) not in (2, 3):
                    print("  usage: drillsetcal <contact_angle> "
                          "[deepest_angle]")
                else:
                    contact = int(tok[1])
                    deepest = int(tok[2]) if len(tok) == 3 else None
                    result = fc.set_drill_calibration(contact, deepest)
                    print(
                        f"  trigger map set: contact="
                        f"{result['contact_angle']}, "
                        f"deepest={result['deepest_angle']}, "
                        f"span={result['span']}"
                    )
                    if result.get("log_path"):
                        plot_path = plot_drill_calibration(
                            result["log_path"], show=False
                        )
                        print(f"  calibrated plot -> {plot_path}")
            elif cmd == "drill" and len(tok) == 2:
                level = float(tok[1])
                target = fc.set_drill_level(level)
                print(f"  drill trigger -> {level:g}% (index angle {target})")
            elif cmd == "drill":
                print("  usage: drill <level_percent 0-100>")
            elif cmd == "drillrelease" and len(tok) == 1:
                target = fc.release_drill_trigger()
                print(f"  drill trigger released (index angle {target})")
            elif cmd == "drillrelease":
                print("  usage: drillrelease")
            elif cmd == "drillplot":
                if len(tok) > 2:
                    print("  usage: drillplot [path]")
                else:
                    plot_drill_calibration(
                        tok[1] if len(tok) == 2 else None
                    )
            elif cmd == "status":
                with fc.io_lock:
                    texts = fc.hand.get_status(as_text=True)
                for n, s in zip(FINGER_NAMES, texts):
                    print(f"  {n:10s}: {s}")
            elif cmd == "angles":
                with fc.io_lock:
                    print(" ", dict(zip(FINGER_NAMES, fc.hand.get_angles())))
            elif cmd == "forces":
                with fc.io_lock:
                    print(" ", dict(zip(FINGER_NAMES, fc.hand.get_forces())), "g")
            elif cmd == "speeds":
                with fc.io_lock:
                    print(" ", dict(zip(FINGER_NAMES,
                                        fc.hand.t.read_shorts("SPEED_SET"))))
            elif cmd == "log":
                sub = tok[1].lower() if len(tok) > 1 else ""
                if sub == "start":
                    print("  logging ->", fc.start_log(tok[2] if len(tok) > 2 else None))
                elif sub == "stop":
                    print("  stopped ->", fc.stop_log())
                else:
                    print("  usage: log start [path] | log stop")
            elif cmd == "mark":
                label = " ".join(tok[1:]) if len(tok) > 1 else None
                result = fc.mark_log(label)
                print(
                    f"  mark {result['mark_id']} at "
                    f"t={result['elapsed_s']:.3f} s: "
                    f"{result['mark_label']}"
                )
                print(f"  HDF5 log -> {result['snapshot_h5']}")
                print(f"  tactile plot -> {result['tactile_plot']}")
            elif cmd == "plottactile":
                if len(tok) > 2:
                    print("  usage: plottactile [path]")
                else:
                    output_path = tok[1] if len(tok) > 1 else None
                    result = fc.plottactile(output_path)
                    print(f"  tactile HDF5 -> {result['h5_path']}")
                    print(f"  tactile plot -> {result['png_path']}")
            elif cmd == "exportmarks":
                if len(tok) > 3:
                    print("  usage: exportmarks [session_or_hand_h5] "
                          "[output.h5]")
                else:
                    session = tok[1] if len(tok) > 1 else None
                    output = tok[2] if len(tok) > 2 else None
                    export_marked_data(session, output)
            elif cmd == "logseq":
                # logseq index 300 middle 300 thumb 400
                args = tok[1:]
                if len(args) < 2 or len(args) % 2 != 0:
                    print("  usage: logseq <finger> <angle> [<finger> <angle> ...]")
                else:
                    steps = [(args[k], int(args[k + 1]))
                             for k in range(0, len(args), 2)]
                    for name, ang in fc.move_sequence(steps, log=True):
                        print(f"  moved {name} -> {ang}")
            elif cmd == "plot":
                plot_log(tok[1] if len(tok) > 1 else None)
            elif cmd == "tactilevideo":
                if len(tok) > 3:
                    print("  usage: tactilevideo [path] [fps]")
                else:
                    video_path = tok[1] if len(tok) > 1 else None
                    video_fps = float(tok[2]) if len(tok) > 2 else 5.0
                    export_tactile_video(video_path, fps=video_fps)
            elif cmd == "speed" and len(tok) == 3:
                fc.set_finger_speed(tok[1], int(tok[2]))
            elif cmd == "speedall" and len(tok) == 2:
                v = int(tok[1])
                if not (0 <= v <= 1000):
                    print("  speed must be 0-1000")
                else:
                    with fc.io_lock:
                        fc.hand.set_speeds([v] * 6)
                    print(f"  all fingers speed -> {v}")
            elif cmd == "force" and len(tok) == 3:
                fc.set_finger_force(tok[1], int(tok[2]))
                print(f"  {tok[1]} force limit -> {tok[2]} g")
            elif cmd == "forceall" and len(tok) == 2:
                v = int(tok[1])
                if not (0 <= v <= 3000):
                    print("  force limit must be 0-3000 g")
                else:
                    with fc.io_lock:
                        fc.hand.set_forces([v] * 6)
                    print(f"  all fingers force limit -> {v} g")
            elif cmd == "openall":
                fc.hand.open_hand()
            elif cmd == "closeall":
                fc.hand.close_hand()
            elif cmd == "wave":
                fc.wave_test()
            elif cmd == "group":
                # group index 300 middle 300  -> move both at once, then hold
                args = tok[1:]
                if len(args) < 2 or len(args) % 2 != 0:
                    print("  usage: group <finger> <angle> "
                          "[<finger> <angle> ...]")
                else:
                    mapping = {args[k]: int(args[k + 1])
                               for k in range(0, len(args), 2)}
                    fc.move_fingers(mapping)   # non-blocking: holds until next cmd
                    print(f"  moving together: {mapping}")
            elif cmd == "seq":
                # seq index 1000 delay 1000 index 500 delay 1000 index 0
                args = tok[1:]
                if len(args) < 2 or len(args) % 2 != 0:
                    print("  usage: seq <finger> <angle> "
                          "[delay <milliseconds> | <finger> <angle> ...]")
                else:
                    actions = [(args[k], args[k + 1])
                               for k in range(0, len(args), 2)]
                    for result in fc.move_timed_sequence(actions):
                        if result[0] == "delay":
                            print(f"  delayed {result[1]} ms")
                        else:
                            print(f"  moved {result[1]} -> {result[2]}")
            elif cmd == "all" and len(tok) == 2:
                with fc.io_lock:
                    fc.hand.set_angles([int(tok[1])] * 6)
            elif cmd == "bend" and len(tok) == 2:
                fc.bend(tok[1])
            elif cmd == "open" and len(tok) == 2:
                fc.open(tok[1])
            elif cmd == "state" and len(tok) == 2:
                for k, v in fc.finger_state(tok[1]).items():
                    print(f"  {k:10s}: {v}")
            elif len(tok) == 2:                    # <finger> <angle>
                fc.move(tok[0], int(tok[1]))
            elif len(tok) == 3:                    # <finger> <angle> <force>
                fc.move(tok[0], int(tok[1]), force=int(tok[2]))
            else:
                print("  ? unknown command — type 'help'")
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    # --- choose ONE transport ---
    hand = RH56Hand(RH56ModbusTCP("192.168.11.210", port=6000))   # Ethernet
    # hand = RH56Hand(RH56Serial("COM3", hand_id=1))              # RS485

    fc = FingerController(hand)
    print("connected. actual angles:",
          dict(zip(FINGER_NAMES, hand.get_angles())))
    try:
        cli(fc)
    finally:
        fc.stop_log()          # flush/close HDF5 if logging was left on
        hand.t.close()
    print("bye")