"""
Stepper control for a Snapmaker linear module (lead screw) on a Raspberry Pi Pico.
MicroPython — external Step/Dir driver (A4988 / DRV8825 / TMC2209 legacy mode).

v2 changes:
  * ramp is now computed in VELOCITY space (v = sqrt(v0^2 + 2ax)) instead of
    interpolating the step delay linearly — the old version spent most of the
    ramp crawling and then jumped to full speed at the very end
  * ramp delays are pre-computed once per (speed, accel) pair and cached
  * loop overhead is measured and subtracted, so commanded speed matches actual
  * SPEED_MAX / ACCEL_MM_S2 are read at call time, so M203/M201 take effect live
  * soft limits moved into move_steps(), so every move path is protected

v3: HX711 load cell sampled on the second core (LoadCell / loadcell()), so a
    blocking move on core 0 never stops the force data. The same core also
    records the stepper position at 50 Hz, so moves are logged step by step.
    Samples are buffered with their ticks_us() time and handed to the PC by
    LoadCell.drain().

Wiring:
    EN   -> GP28   (driver enable, active LOW on most carriers)
    STEP -> GP2
    DIR  -> GP3
    LIMIT-> GP27   (switch to GND, internal pull-up)
    HX711 DT  -> GP17
    HX711 SCK -> GP16
"""

from machine import Pin
from array import array
import _thread
import machine
import math
import time

# ---------------------------------------------------------------- configuration
PIN_EN = 28
PIN_STEP = 2
PIN_DIR = 3
PIN_LIMIT = 27

ENABLE_ACTIVE_LOW = True
LIMIT_ACTIVE_LOW = True
INVERT_DIR = True

# Mechanics — 8 mm lead confirmed for Snapmaker A250/A350 X/Y/Z modules.
# NOTE: the TMC2209 (A250T/A350T/F-series) X and Y modules use a 20 mm lead —
# if yours is one of those, LEAD_MM must be 20.0 or every distance is 2.5x off.
FULL_STEPS_PER_REV = 200
MICROSTEPS = 16             # lowering this raises the achievable top speed
LEAD_MM = 8.0
STEPS_PER_MM = FULL_STEPS_PER_REV * MICROSTEPS / LEAD_MM

TRAVEL_MM = 120.0
HOME_DIR = -1
BACKOFF_MM = 3.0

# Speeds in mm/s
SPEED_START = 3.0           # ramp start/end speed — must not stall
SPEED_MAX = 10
SPEED_HOME_FAST = 12.0
SPEED_HOME_SLOW = 2.0
ACCEL_MM_S2 = 300.0

STEP_PULSE_US = 3
DIR_SETUP_US = 5

# Time the Python loop itself burns per step, subtracted from every delay.
# Run Stepper().calibrate_overhead() once and paste the number here.
LOOP_OVERHEAD_US = 0

_MAX_RAMP_STEPS = 6000      # memory guard on the pre-computed table

# HX711 load cell
PIN_HX_DT = 17
PIN_HX_SCK = 16
LC_BUFFER = 16384           # samples: ~200 s at 80 Hz, ~27 min at 10 Hz
GUIDE_SAMPLE_US = 20000     # stepper position recorded every 20 ms (50 Hz)


# --------------------------------------------------------------- ramp table
_ramp_cache = {}


def ramp_table(v_max, accel):
    """Step delays (us) for accelerating from SPEED_START to v_max at `accel`."""
    key = (int(v_max * 100), int(accel * 10))
    tbl = _ramp_cache.get(key)
    if tbl is not None:
        return tbl

    dist = (v_max ** 2 - SPEED_START ** 2) / (2.0 * accel)      # mm of ramp
    n = max(1, min(int(dist * STEPS_PER_MM), _MAX_RAMP_STEPS))
    tbl = array("I", bytearray(4 * n))
    for i in range(n):
        x = (i + 0.5) / STEPS_PER_MM
        v = math.sqrt(SPEED_START ** 2 + 2.0 * accel * x)
        if v > v_max:
            v = v_max
        d = int(1000000 / (v * STEPS_PER_MM)) - LOOP_OVERHEAD_US
        tbl[i] = d if d > 0 else 0

    if len(_ramp_cache) > 6:            # keep RAM bounded
        _ramp_cache.clear()
    _ramp_cache[key] = tbl
    return tbl


# ---------------------------------------------------------------------- driver
class Stepper:
    def __init__(self):
        self.en = Pin(PIN_EN, Pin.OUT, value=1 if ENABLE_ACTIVE_LOW else 0)
        self.step = Pin(PIN_STEP, Pin.OUT, value=0)
        self.dir = Pin(PIN_DIR, Pin.OUT, value=0)
        self.limit = Pin(PIN_LIMIT, Pin.IN,
                         Pin.PULL_UP if LIMIT_ACTIVE_LOW else Pin.PULL_DOWN)
        self.position = 0
        self.homed = False

    # --- low level -----------------------------------------------------------
    def enable(self, on=True):
        self.en.value((0 if on else 1) if ENABLE_ACTIVE_LOW else (1 if on else 0))
        time.sleep_ms(5)

    def limit_hit(self):
        v = self.limit.value()
        return v == 0 if LIMIT_ACTIVE_LOW else v == 1

    def _set_dir(self, sign):
        forward = sign > 0
        if INVERT_DIR:
            forward = not forward
        self.dir.value(1 if forward else 0)
        time.sleep_us(DIR_SETUP_US)

    @staticmethod
    def _delay_us(speed_mm_s):
        d = int(1000000 / (speed_mm_s * STEPS_PER_MM)) - LOOP_OVERHEAD_US
        return d if d > 0 else 0

    # --- motion --------------------------------------------------------------
    def move_steps(self, steps, speed=None, ignore_limit=False, homing=False):
        """Blocking move with a trapezoidal (or triangular) ramp."""
        if steps == 0:
            return 0
        if speed is None or speed > SPEED_MAX:
            speed = SPEED_MAX

        # soft limits — every move path goes through here
        if self.homed:
            hi = int(TRAVEL_MM * STEPS_PER_MM)
            target = self.position + steps
            if target < 0 or target > hi:
                clamped = 0 if target < 0 else hi
                print("// soft limit: {:.2f} -> {:.2f} mm".format(
                    target / STEPS_PER_MM, clamped / STEPS_PER_MM))
                steps = clamped - self.position
                if steps == 0:
                    return 0
        elif not homing and abs(steps) > 5 * STEPS_PER_MM:
            raise RuntimeError("not homed — jog limited to 5 mm (or run home())")

        sign = 1 if steps > 0 else -1
        n = abs(steps)
        self._set_dir(sign)

        d_min = self._delay_us(speed)
        if speed <= SPEED_START:
            tbl, ramp_n = None, 0            # too slow to need a ramp
        else:
            tbl = ramp_table(speed, ACCEL_MM_S2)
            ramp_n = min(len(tbl), n // 2)   # short move -> triangular profile

        stp = self.step                      # local refs = fewer lookups per step
        sleep_us = time.sleep_us
        pulse = STEP_PULSE_US
        check = (not ignore_limit) and sign == HOME_DIR
        done = 0

        for i in range(n):
            if check and self.limit_hit():
                print("limit switch hit — move aborted")
                break

            if i < ramp_n:
                d = tbl[i]
            elif i >= n - ramp_n:
                d = tbl[n - 1 - i]
            else:
                d = d_min

            stp.value(1)
            sleep_us(pulse)
            stp.value(0)
            if d > pulse:
                sleep_us(d - pulse)

            self.position += sign
            done += 1

        return done * sign

    def move_mm(self, mm, speed=None):
        return self.move_steps(int(round(mm * STEPS_PER_MM)), speed) / STEPS_PER_MM

    def move_to_mm(self, target_mm, speed=None):
        if not self.homed:
            raise RuntimeError("home() first — absolute position is unknown")
        delta = int(round(target_mm * STEPS_PER_MM)) - self.position
        return self.move_steps(delta, speed)

    def get_mm(self):
        return self.position / STEPS_PER_MM

    # --- homing --------------------------------------------------------------
    def home(self):
        self.enable(True)
        self.homed = False
        max_steps = int((TRAVEL_MM + 10) * STEPS_PER_MM)

        if not self.limit_hit():
            self.move_steps(HOME_DIR * max_steps, SPEED_HOME_FAST, homing=True)
            if not self.limit_hit():
                raise RuntimeError("homing failed: switch never triggered")

        self.move_steps(int(-HOME_DIR * BACKOFF_MM * STEPS_PER_MM),
                        SPEED_HOME_SLOW, ignore_limit=True, homing=True)
        if self.limit_hit():
            raise RuntimeError("switch still closed after back-off — check wiring")

        self.move_steps(HOME_DIR * int((BACKOFF_MM + 2) * STEPS_PER_MM),
                        SPEED_HOME_SLOW, homing=True)
        if not self.limit_hit():
            raise RuntimeError("homing failed on second approach")

        self.position = 0
        self.homed = True
        print("homed, position = 0.0 mm")

    # --- diagnostics ---------------------------------------------------------
    def calibrate_overhead(self, n=20000):
        """Pulse with no commanded delay to measure the Python loop cost.
        The driver is disabled first, so nothing moves. Paste the printed value
        into LOOP_OVERHEAD_US and reboot."""
        self.enable(False)
        stp = self.step
        sleep_us = time.sleep_us
        t0 = time.ticks_us()
        for _ in range(n):
            stp.value(1)
            sleep_us(STEP_PULSE_US)
            stp.value(0)
        dt = time.ticks_diff(time.ticks_us(), t0)
        per = dt / n
        print("loop cost {:.1f} us/step -> ceiling {:.0f} steps/s = {:.1f} mm/s".format(
            per, 1000000 / per, 1000000 / per / STEPS_PER_MM))
        print("set LOOP_OVERHEAD_US = {}".format(int(per - STEP_PULSE_US)))
        return per

    def bench(self, low=5.0, high=105.0, speeds=(10, 20, 30, 40, 50, 60)):
        """Commanded vs actual speed, then check for lost steps."""
        global SPEED_MAX
        keep = SPEED_MAX
        SPEED_MAX = max(SPEED_MAX, max(speeds))
        span = high - low
        try:
            self.home()
            for v in speeds:
                self.move_to_mm(low, SPEED_HOME_FAST)
                t0 = time.ticks_us()
                self.move_to_mm(high, v)
                dt = time.ticks_diff(time.ticks_us(), t0) / 1e6
                print("cmd {:>4.1f} mm/s -> actual {:>5.1f} mm/s".format(v, span / dt))
            before = self.position
            self.home()
            print("drift after run: {:.3f} mm".format(
                (before - int(low * STEPS_PER_MM)) / STEPS_PER_MM))
        finally:
            SPEED_MAX = keep


# ------------------------------------------------------------------ load cell
class LoadCell:
    """HX711 (channel A, gain 128) and stepper position, read on core 1.

    Every HX711 reading is stored with the ticks_us() time its conversion
    became ready, and every 20 ms the stepper's step count and homed flag are
    stored with theirs, in two ring buffers the PC empties with drain(). The
    loop allocates nothing, so the garbage collector on core 0 never has to
    stop it, and the 25 clock pulses run with interrupts off so SCK is never
    held high long enough (>60 us) to power the HX711 down.
    """

    def __init__(self, dt=PIN_HX_DT, sck=PIN_HX_SCK, size=LC_BUFFER):
        self.dt = Pin(dt, Pin.IN, Pin.PULL_UP)
        self.sck = Pin(sck, Pin.OUT, value=0)
        self.size = size
        self.t = array("I", bytearray(4 * size))     # ticks_us per sample
        self.v = array("i", bytearray(4 * size))     # signed raw counts
        self.gt = array("I", bytearray(4 * size))    # ticks_us per position
        self.gp = array("i", bytearray(4 * size))    # stepper.position, steps
        self.gh = bytearray(size)                    # stepper.homed
        # every attribute exists before the thread starts: the two cores
        # must never resize the instance's attribute table at the same time
        self.stepper = None
        self.head = 0           # samples written (total)
        self.tail = 0           # samples handed to the PC (total)
        self.dropped = 0
        self.ghead = 0          # same three for positions
        self.gtail = 0
        self.gdropped = 0
        self.running = False
        self.stopped = True

    def _run(self):
        dt, sck = self.dt, self.sck
        t, v, size = self.t, self.v, self.size
        gt, gp, gh = self.gt, self.gp, self.gh
        ticks = time.ticks_us
        diff, add = time.ticks_diff, time.ticks_add
        sleep_us = time.sleep_us
        next_guide = ticks()
        while self.running:
            now = ticks()
            if diff(now, next_guide) >= 0:
                stepper = self.stepper
                if stepper is not None:
                    j = self.ghead % size
                    gt[j] = now
                    gp[j] = stepper.position
                    gh[j] = 1 if stepper.homed else 0
                    self.ghead += 1
                next_guide = add(next_guide, GUIDE_SAMPLE_US)
                if diff(now, next_guide) >= 0:      # fell behind: resync
                    next_guide = add(now, GUIDE_SAMPLE_US)
            if dt.value():                  # conversion not ready yet
                sleep_us(100)
                continue
            stamp = ticks()
            irq = machine.disable_irq()
            raw = 0
            for _ in range(24):
                sck.value(1)
                raw = (raw << 1) | dt.value()
                sck.value(0)
            sck.value(1)                    # 25th pulse: channel A, gain 128
            sck.value(0)
            machine.enable_irq(irq)
            if raw & 0x800000:
                raw -= 0x1000000
            i = self.head % size
            t[i] = stamp
            v[i] = raw
            self.head += 1                  # publish only after the slot is full
        self.stopped = True

    def start(self):
        self.tail = self.head               # forget samples nobody asked for
        self.gtail = self.ghead
        if not self.running:
            self.running = True
            self.stopped = False
            _thread.start_new_thread(self._run, ())
        return self

    def stop(self, timeout_ms=500):
        self.running = False
        t0 = time.ticks_ms()
        while not self.stopped and time.ticks_diff(
                time.ticks_ms(), t0) < timeout_ms:
            time.sleep_ms(5)
        return self.stopped

    def drain(self, limit=200):
        """Print up to `limit` buffered samples and positions for the PC.

        __LC__|<ticks_us now>|<dropped>|<samples left>|<g dropped>|<g left>
        <ticks>,<raw>;<ticks>,<raw>;...
        <ticks>,<steps>,<homed>;...
        """
        head, ghead = self.head, self.ghead     # read before `now`: nothing
        now = time.ticks_us()                   # is newer than the header
        size = self.size
        tail = self.tail
        if head - tail > size:              # PC fell behind: oldest are gone
            self.dropped += head - tail - size
            tail = head - size
        n = min(head - tail, limit)
        parts = []
        for k in range(tail, tail + n):
            i = k % size
            parts.append("%d,%d" % (self.t[i], self.v[i]))
        self.tail = tail + n
        gtail = self.gtail
        if ghead - gtail > size:
            self.gdropped += ghead - gtail - size
            gtail = ghead - size
        n = min(ghead - gtail, limit)
        gparts = []
        for k in range(gtail, gtail + n):
            i = k % size
            gparts.append("%d,%d,%d" % (self.gt[i], self.gp[i], self.gh[i]))
        self.gtail = gtail + n
        print("__LC__|%d|%d|%d|%d|%d" % (now, self.dropped, head - self.tail,
                                         self.gdropped, ghead - self.gtail))
        print(";".join(parts))
        print(";".join(gparts))


_loadcell = None


def loadcell(stepper=None):
    """The one LoadCell (core 1 can run only one thread), started and
    recording `stepper`'s position."""
    global _loadcell
    if _loadcell is None:
        _loadcell = LoadCell()
    _loadcell.stepper = stepper
    return _loadcell.start()


# ------------------------------------------------------------------------ demo
if __name__ == "__main__":
    m = Stepper()
    try:
        m.enable(True)
        m.home()
        for target in (0, 0, 0):
            m.move_to_mm(target)
            print("at {:.2f} mm".format(m.get_mm()))
            time.sleep(0.5)
        m.move_to_mm(0.0)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        m.enable(False)