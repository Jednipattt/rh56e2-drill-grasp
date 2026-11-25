"""
RH56E2 Dexterous Hand Driver (Inspire Robots)
=============================================
Supports two transports:
  1. RS485 serial   — proprietary frame protocol (0xEB 0x90 ...), per PRJ-02-TS-U-001/010
  2. Modbus TCP     — RH56DFTP/E2 only (default IP 192.168.11.210, port 6000)

DOF index convention (m = 0..5):
  0 = little finger, 1 = ring finger, 2 = middle finger,
  3 = index finger,  4 = thumb bending, 5 = thumb rotation

Value conventions:
  ANGLE  : 0..1000  (0 = fully bent, 1000 = fully open, -1 = no action)
  SPEED  : 0..1000
  FORCE  : threshold in grams (RH56: 0..1000, E2/DFTP: 0..3000)
  STATUS : 0 unclench, 1 grasping, 2 reached position,
           3 reached force threshold, 5 current protection,
           6 locked-rotor stop, 7 actuator fault

Dependencies:
  pip install pyserial pymodbus
"""

import struct
import time

# ---------------------------------------------------------------------------
# Register map (byte addresses, from the user manuals)
# ---------------------------------------------------------------------------
REG = {
    "HAND_ID":        1000,
    "REDU_RATIO":     1002,
    "CLEAR_ERROR":    1004,
    "SAVE":           1005,
    "RESET_PARA":     1006,
    "FORCE_CLB":      1009,   # force sensor calibration (no-load required!)
    "DEFAULT_SPEED":  1032,
    "DEFAULT_FORCE":  1044,
    "POS_SET":        1474,
    "ANGLE_SET":      1486,
    "FORCE_SET":      1498,
    "SPEED_SET":      1522,
    "POS_ACT":        1534,
    "ANGLE_ACT":      1546,
    "FORCE_ACT":      1582,
    "CURRENT":        1594,
    "ERROR":          1606,
    "STATUS":         1612,
    "TEMP":           1618,
    # Not in the manual -- found by sniffing the vendor software.
    # One byte per DOF: 0 position, 1 force.
    "CONTROL_MODE":   1625,
}

# Tactile sensor blocks (RH56E2/DFTP only): (byte_address, byte_length, rows, cols)
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

STATUS_TEXT = {
    0: "unclenching", 1: "grasping", 2: "reached target position",
    3: "reached force threshold", 5: "stopped: current protection",
    6: "stopped: locked rotor", 7: "stopped: actuator fault",
}

FINGER_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


def _shorts_to_bytes(values):
    """List of signed shorts -> little-endian bytes (low byte first)."""
    return b"".join(struct.pack("<h", int(v)) for v in values)


def _bytes_to_shorts(data):
    return list(struct.unpack("<" + "h" * (len(data) // 2), data))


# ===========================================================================
# Transport 1: RS485 serial (proprietary 0xEB 0x90 protocol)
# ===========================================================================
class RH56Serial:
    """RS485 driver. Params: 115200 bps, 8N1 (per manual)."""

    def __init__(self, port, hand_id=1, baudrate=115200, timeout=0.5):
        import serial  # pyserial
        self.hand_id = hand_id
        self.ser = serial.Serial(port=port, baudrate=baudrate,
                                 bytesize=8, parity="N", stopbits=1,
                                 timeout=timeout)

    def close(self):
        self.ser.close()

    # ---- low-level frame I/O -------------------------------------------
    @staticmethod
    def _checksum(frame_wo_header):
        return sum(frame_wo_header) & 0xFF

    def _read_registers(self, address, nbytes):
        """Read `nbytes` starting at byte `address`. Returns raw bytes."""
        frame = bytearray([0xEB, 0x90, self.hand_id, 0x04, 0x11,
                           address & 0xFF, (address >> 8) & 0xFF, nbytes])
        frame.append(self._checksum(frame[2:]))
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        expected = 8 + nbytes + 1  # header(2)+id+len+flag+addr(2)+data+cksum
        resp = self.ser.read(expected)
        if len(resp) < expected or resp[0] != 0x90 or resp[1] != 0xEB:
            raise IOError(f"bad/short response ({len(resp)}/{expected} bytes)")
        if self._checksum(resp[2:-1]) != resp[-1]:
            raise IOError("checksum mismatch in response")
        return bytes(resp[7:7 + nbytes])

    def _write_registers(self, address, data):
        """Write raw bytes `data` starting at byte `address`."""
        n = len(data)
        frame = bytearray([0xEB, 0x90, self.hand_id, n + 3, 0x12,
                           address & 0xFF, (address >> 8) & 0xFF])
        frame += data
        frame.append(self._checksum(frame[2:]))
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        resp = self.ser.read(9)  # ack frame
        if len(resp) < 9 or resp[0] != 0x90 or resp[1] != 0xEB:
            raise IOError("no/short write acknowledgment")
        return True

    # ---- typed helpers --------------------------------------------------
    def read_shorts(self, name, count=6):
        return _bytes_to_shorts(self._read_registers(REG[name], count * 2))

    def read_bytes(self, name, count=6):
        return list(self._read_registers(REG[name], count))

    def write_shorts(self, name, values):
        return self._write_registers(REG[name], _shorts_to_bytes(values))

    def write_short_at(self, name, index, value):
        """Write one short of a block, leaving the other DOFs untouched."""
        return self._write_registers(REG[name] + 2 * index,
                                     _shorts_to_bytes([value]))

    def write_byte(self, name, value):
        return self._write_registers(REG[name], bytes([value & 0xFF]))


# ===========================================================================
# Transport 2: Modbus TCP (RH56E2 / RH56DFTP)
# ===========================================================================
class RH56ModbusTCP:
    """
    Modbus TCP driver for the E2/DFTP variant.
    Default IP 192.168.11.210, port 6000 (per manual).

    NOTE on addressing: this follows the common Inspire SDK convention of
    using the documented byte address directly as the Modbus register
    address, one 16-bit register per short. If reads look shifted on your
    firmware, try address // 2 — verify once against ANGLE_ACT.
    """

    def __init__(self, host="192.168.11.210", port=6000, unit=1):
        from pymodbus.client import ModbusTcpClient
        self.client = ModbusTcpClient(host=host, port=port)
        self.unit = unit
        if not self.client.connect():
            raise IOError(f"cannot connect to {host}:{port}")

    def close(self):
        self.client.close()

    def read_shorts(self, name, count=6):
        rr = self.client.read_holding_registers(REG[name], count=count,
                                                device_id=self.unit)
        if rr.isError():
            raise IOError(f"modbus read error: {rr}")
        # convert unsigned register values to signed shorts
        return [v - 65536 if v > 32767 else v for v in rr.registers]

    def read_bytes(self, name, count=6):
        # byte registers are packed 2-per-register (lo, hi)
        nregs = (count + 1) // 2
        rr = self.client.read_holding_registers(REG[name], count=nregs,
                                                device_id=self.unit)
        if rr.isError():
            raise IOError(f"modbus read error: {rr}")
        out = []
        for v in rr.registers:
            out += [v & 0xFF, (v >> 8) & 0xFF]
        return out[:count]

    def write_shorts(self, name, values):
        regs = [v & 0xFFFF for v in values]  # two's complement for -1
        rq = self.client.write_registers(REG[name], regs, device_id=self.unit)
        if rq.isError():
            raise IOError(f"modbus write error: {rq}")
        return True

    def write_short_at(self, name, index, value):
        """Write one short of a block, leaving the other DOFs untouched."""
        # byte addressing: short `index` of a block starts 2 * index bytes in
        rq = self.client.write_registers(REG[name] + 2 * index,
                                         [value & 0xFFFF], device_id=self.unit)
        if rq.isError():
            raise IOError(f"modbus write error: {rq}")
        return True

    def write_byte(self, name, value):
        rq = self.client.write_register(REG[name], value & 0xFF,
                                        device_id=self.unit)
        if rq.isError():
            raise IOError(f"modbus write error: {rq}")
        return True

    def read_tactile(self, region):
        """Read one tactile block, e.g. 'index_tip'. Returns 2D list (rows x cols)."""
        addr, nbytes, rows, cols = TACTILE[region]
        values = []
        remaining = nbytes // 2  # 16-bit points
        offset = 0
        while remaining > 0:
            chunk = min(remaining, 120)  # stay under Modbus 125-reg limit
            rr = self.client.read_holding_registers(addr + offset * 2,
                                                    count=chunk,
                                                    device_id=self.unit)
            if rr.isError():
                raise IOError(f"modbus read error: {rr}")
            values += rr.registers
            offset += chunk
            remaining -= chunk
        return [values[r * cols:(r + 1) * cols] for r in range(rows)]


# ===========================================================================
# High-level hand API (transport-agnostic)
# ===========================================================================
class RH56Hand:
    """
    High-level control wrapper. Pass either transport:

        hand = RH56Hand(RH56Serial("/dev/ttyUSB0", hand_id=1))
        hand = RH56Hand(RH56ModbusTCP("192.168.11.210"))
    """

    def __init__(self, transport):
        self.t = transport

    # ---- basic motion ----------------------------------------------------
    def set_angles(self, angles):
        """angles: 6 values, 0..1000 or -1 (no action)."""
        assert len(angles) == 6
        return self.t.write_shorts("ANGLE_SET", angles)

    def set_speeds(self, speeds):
        assert len(speeds) == 6
        return self.t.write_shorts("SPEED_SET", speeds)

    def set_forces(self, forces_g):
        """Force thresholds in grams per DOF (fingertip-referenced!)."""
        assert len(forces_g) == 6
        return self.t.write_shorts("FORCE_SET", forces_g)

    def set_force(self, index, force_g):
        """Force threshold of ONE DOF (0..5); the other five are not written."""
        assert 0 <= index <= 5
        return self.t.write_short_at("FORCE_SET", index, force_g)

    def get_control_modes(self):
        """Control mode per DOF: 0 = position mode, 1 = force mode."""
        return self.t.read_bytes("CONTROL_MODE")

    def set_control_modes(self, modes):
        """modes: 6 values, 0 = position mode, 1 = force mode."""
        assert len(modes) == 6
        m = [1 if v else 0 for v in modes]
        # byte registers travel two per short, low byte = lower address
        return self.t.write_shorts(
            "CONTROL_MODE", [m[i] | (m[i + 1] << 8) for i in (0, 2, 4)])

    def open_hand(self, speed=500):
        self.set_speeds([speed] * 6)
        return self.set_angles([1000] * 6)

    def close_hand(self, speed=300):
        self.set_speeds([speed] * 6)
        return self.set_angles([0] * 6)

    # ---- state -----------------------------------------------------------
    def get_angles(self):
        return self.t.read_shorts("ANGLE_ACT")

    def get_forces(self):
        """Actual contact force per finger (grams). E2 range -4000..4000."""
        return self.t.read_shorts("FORCE_ACT")

    def get_currents(self):
        return self.t.read_shorts("CURRENT")

    def get_status(self, as_text=False):
        s = self.t.read_bytes("STATUS")
        return [STATUS_TEXT.get(v, f"code {v}") for v in s] if as_text else s

    def get_errors(self):
        return self.t.read_bytes("ERROR")

    def get_temperatures(self):
        return self.t.read_bytes("TEMP")

    # ---- maintenance -----------------------------------------------------
    def clear_errors(self):
        return self.t.write_byte("CLEAR_ERROR", 1)

    def calibrate_force_sensors(self):
        """Takes ~6 s. Hand MUST be in no-load state (touching nothing)."""
        self.t.write_byte("FORCE_CLB", 1)
        time.sleep(7)

    def save_to_flash(self):
        return self.t.write_byte("SAVE", 1)

    # ---- grasp-level helpers ----------------------------------------------
    def wait_until_stopped(self, timeout=5.0, poll=0.05):
        """Block until every commanded DOF stops (status 2/3/5/6/7)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.get_status()
            if all(s in (2, 3, 5, 6, 7) for s in st):
                return st
            time.sleep(poll)
        return self.get_status()

    def grasp(self, angles, forces_g, speed=300, timeout=5.0):
        """
        Force-adaptive grasp:
        set thresholds first, then command angles, then wait.
        Returns dict with final status and measured contact forces.
        """
        self.set_forces(forces_g)
        self.set_speeds([speed] * 6)
        time.sleep(0.02)
        self.set_angles(angles)
        status = self.wait_until_stopped(timeout=timeout)
        return {
            "status": status,
            "status_text": [STATUS_TEXT.get(s, str(s)) for s in status],
            "angles_act": self.get_angles(),
            "forces_act": self.get_forces(),
            "contact": [s == 3 for s in status],  # True where force threshold hit
        }


# ===========================================================================
# Example grasp taxonomy for the project (tune angles on real hardware!)
# ===========================================================================
#            [little, ring, middle, index, th_bend, th_rot]
GRASP_LIBRARY = {
    # fragile precision grasps (egg, light bulb)
    "tripod":      {"angles": [1000, 1000, 400, 400, 500, 200],
                    "force":  [100, 100, 150, 150, 150, 150]},
    "precision":   {"angles": [1000, 800, 500, 450, 500, 250],
                    "force":  [150, 150, 200, 200, 200, 200]},
    # compliant sphere (tennis ball)
    "sphere":      {"angles": [450, 450, 450, 450, 450, 400],
                    "force":  [300, 300, 300, 300, 350, 300]},
    # cylindrical objects (glass cup, medicine bottle)
    "prismatic":   {"angles": [500, 500, 500, 500, 550, 500],
                    "force":  [250, 250, 250, 250, 300, 250]},
    "medium_wrap": {"angles": [300, 300, 300, 300, 350, 600],
                    "force":  [400, 400, 400, 400, 450, 400]},
    # tool-like object (hand drill)
    "light_tool":  {"angles": [200, 200, 250, 300, 300, 700],
                    "force":  [600, 600, 600, 600, 700, 600]},
    "power":       {"angles": [100, 100, 100, 100, 200, 800],
                    "force":  [800, 800, 800, 800, 900, 800]},
}

# YOLO class -> (grasp type, description) for the six project objects
OBJECT_TO_GRASP = {
    "egg":             "tripod",
    "light_bulb":      "precision",
    "tennis_ball":     "sphere",
    "glass_cup":       "prismatic",
    "medicine_bottle": "prismatic",
    "hand_drill":      "light_tool",
}


def grasp_detected_object(hand, yolo_class):
    """Full object-aware grasp: YOLO class -> taxonomy -> force-adaptive execution."""
    grasp_name = OBJECT_TO_GRASP[yolo_class]
    g = GRASP_LIBRARY[grasp_name]
    print(f"[{yolo_class}] -> {grasp_name} grasp, forces {g['force']}")
    hand.open_hand()
    hand.wait_until_stopped()
    result = hand.grasp(g["angles"], g["force"])
    print("  status :", result["status_text"])
    print("  forces :", result["forces_act"], "g")
    return result


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    # --- choose ONE transport ---
    #hand = RH56Hand(RH56Serial("/dev/ttyUSB0", hand_id=1))          # RS485
    hand = RH56Hand(RH56ModbusTCP("192.168.11.210", port=6000))  # Ethernet

    print("angles :", hand.get_angles())
    print("temps  :", hand.get_temperatures(), "degC")
    print("errors :", hand.get_errors())

    # simple open/close test
    hand.open_hand()
    print("status :", hand.wait_until_stopped())

    # object-aware grasp demo (as if YOLO detected an egg)
    grasp_detected_object(hand, "glass_cup")

    time.sleep(2)
    hand.open_hand()
    hand.t.close()