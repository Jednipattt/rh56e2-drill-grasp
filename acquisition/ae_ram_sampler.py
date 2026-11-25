# ae_ram_sampler.py - Board 2 (MOTION 2350 Pro, RP2350A) AE sampler that runs
# from RAM only. The PC pastes this file into the MicroPython raw REPL (see
# ae_board2.py); it is never written to the board's flash, so a reset always
# brings back the untouched main.py (old AE firmware).
#
# Sampling: the ADC free-runs in round-robin over AIN0 (GP26) and AIN1 (GP27)
# at 2 x FS_HZ, paced by its own 48 MHz clock divider, and a DMA channel in
# ENDLESS mode moves every FIFO result into a 32 KB RAM ring (8192 pairs,
# 409.6 ms). No Python runs per sample. A hard timer IRQ reads the DMA write
# pointer every 50 ms, so the sample counter stays exact even while a USB
# write blocks; the main loop only frames and sends what the counter says
# is there. The sample index is the timebase.
#
# Wire (board -> host), little-endian, one frame per FRAME sample pairs:
#   magic b"\xA5\xE2" | seq u32 | first sample index u64 | n u16 | flags u16
#   | n x (ch0 u16, ch1 u16) | CRC-16/CCITT-FALSE u16 over seq..samples
# flags: bit0 samples were skipped before this frame (overrun),
#        bit1 ADC FIFO overflow seen, bit2 first frame after START.
# Text replies (between frames, one line each): "INFO {json}",
# "STARTED", "STOPPED <frames> <samples> <overruns>", "ERR <msg>".
# Commands (host -> board, one per line): INFO, START, STOP, EXIT.

import array
import sys
import select
import micropython
import rp2
import uctypes
import ustruct as struct
import utime
from machine import ADC, Timer, mem32

VERSION = "ae_ram_sampler/1.0"
FS_HZ = 20000
FRAME = 64                      # sample pairs per frame = 3.2 ms
RING_PAIRS = 8192               # DMA ring: 4 bytes/pair -> 32 KB (DMA max)
PINS = (26, 27)                 # ch0, ch1 (AIN0, AIN1)
MAGIC = b"\xA5\xE2"
HDR = "<2sIQHH"
HDR_LEN = 18
PAYLOAD = FRAME * 4

ADC_BASE = 0x400A0000
ADC_CS, ADC_FCS = ADC_BASE + 0x00, ADC_BASE + 0x08
ADC_FIFO, ADC_DIV = ADC_BASE + 0x0C, ADC_BASE + 0x10
ADC_CLOCK_HZ = 48_000_000
DREQ_ADC = 48
DMA_BASE = 0x50000000
DMA_ENDLESS = 0xF0000001        # RP2350 TRANS_COUNT mode ENDLESS

micropython.alloc_emergency_exception_buf(128)

_RING_BYTES = RING_PAIRS * 4
_RING_BITS = 15                 # 1 << 15 == _RING_BYTES
_backing = bytearray(_RING_BYTES * 2)
_ring_addr = ((uctypes.addressof(_backing) + _RING_BYTES - 1)
              & ~(_RING_BYTES - 1))         # DMA ring must be size-aligned
ring = uctypes.bytearray_at(_ring_addr, _RING_BYTES)
_ring_mv = memoryview(ring)

_frame = bytearray(HDR_LEN + PAYLOAD + 2)
_frame_mv = memoryview(_frame)

_crc_tab = array.array("H", [0] * 256)
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) if _c & 0x8000 else (_c << 1)
    _crc_tab[_i] = _c & 0xFFFF


@micropython.viper
def _crc16(buf: ptr8, start: int, end: int, tab: ptr16) -> int:
    crc = 0xFFFF
    i = start
    while i < end:
        crc = ((crc << 8) & 0xFFFF) ^ tab[((crc >> 8) ^ buf[i]) & 0xFF]
        i += 1
    return crc


# Written only by the timer IRQ: [halfwords written mod 2**29, last offset].
# Kept below 2**30 so the IRQ never allocates a big int.
_hw = array.array("I", [0, 0])
_dma = None
_wa_reg = 0
_timer = None


def _tick(_t):
    off = (mem32[_wa_reg] - _ring_addr) >> 1
    delta = (off - _hw[1]) & (RING_PAIRS * 2 - 1)
    _hw[1] = off
    _hw[0] = (_hw[0] + delta) & 0x1FFFFFFF


class Sampler:
    def __init__(self):
        self.running = False
        self.total_hw = 0       # unwrapped halfwords written by the DMA
        self._last_hw = 0
        self.sent = 0           # sample pairs consumed (framed or skipped)
        self.seq = 0
        self.overruns = 0
        self.fifo_overflows = 0
        self._flags = 0

    def start(self):
        global _dma, _wa_reg, _timer
        self.stop()
        ADC(PINS[0]), ADC(PINS[1])          # put both pads in analog mode
        mem32[ADC_CS] = 0
        mem32[ADC_FCS] = 0
        while not mem32[ADC_FCS] & (1 << 8):    # drain until FIFO EMPTY
            mem32[ADC_FIFO]
        mem32[ADC_FCS] = (1 << 11) | (1 << 10)  # clear OVER/UNDER (W1C)
        divisor = ADC_CLOCK_HZ // (FS_HZ * 2) - 1   # period = 1 + INT cycles
        mem32[ADC_DIV] = divisor << 8
        mem32[ADC_FCS] = 1 | (1 << 3) | (1 << 24)   # EN, DREQ_EN, THRESH=1
        _hw[0] = _hw[1] = 0
        self.total_hw = self._last_hw = self.sent = self.seq = 0
        self.overruns = self.fifo_overflows = 0
        self._flags = 4
        # The DMA is armed before the ADC starts, so halfword 0 is the first
        # conversion and even halfwords are always ch0.
        _dma = rp2.DMA()
        _wa_reg = DMA_BASE + _dma.channel * 0x40 + 0x04
        ctrl = _dma.pack_ctrl(size=1, inc_read=False, inc_write=True,
                              treq_sel=DREQ_ADC, ring_sel=True,
                              ring_size=_RING_BITS)
        _dma.config(read=ADC_FIFO, write=_ring_addr, count=DMA_ENDLESS,
                    ctrl=ctrl, trigger=True)
        _timer = Timer(mode=Timer.PERIODIC, period=50, callback=_tick,
                       hard=True)
        cs = 1 | (0 << 12) | (0x3 << 16)    # EN, AINSEL=0, RROBIN AIN0+AIN1
        mem32[ADC_CS] = cs
        while not mem32[ADC_CS] & (1 << 8):     # READY
            pass
        mem32[ADC_CS] = cs | (1 << 3)           # START_MANY
        self.running = True

    def stop(self):
        global _dma, _timer
        mem32[ADC_CS] = 0
        if _timer is not None:
            _timer.deinit()
            _timer = None
        if _dma is not None:
            try:
                _dma.active(False)
                _dma.close()
            except Exception:
                pass
            _dma = None
        mem32[ADC_FCS] = 0
        self.running = False

    def _written(self):
        hw = _hw[0]
        self.total_hw += (hw - self._last_hw) & 0x1FFFFFFF
        self._last_hw = hw
        return self.total_hw >> 1

    def next_frame(self):
        """One framed bytearray view if FRAME pairs are ready, else None."""
        written = self._written()
        if written - self.sent > RING_PAIRS - 2 * FRAME:
            # The DMA has lapped us (host stopped reading): skip ahead,
            # frame-aligned, and flag the gap in the next frame.
            self.sent = ((written - RING_PAIRS // 2) // FRAME) * FRAME
            self.overruns += 1
            self._flags |= 1
        if written - self.sent < FRAME:
            return None
        if mem32[ADC_FCS] & (1 << 11):
            mem32[ADC_FCS] = 1 | (1 << 3) | (1 << 24) | (1 << 11)
            self.fifo_overflows += 1
            self._flags |= 2
        start = (self.sent % RING_PAIRS) * 4
        _frame_mv[HDR_LEN:HDR_LEN + PAYLOAD] = _ring_mv[start:start + PAYLOAD]
        flags = self._flags
        if self._written() - self.sent > RING_PAIRS - FRAME:
            flags |= 1                      # overwritten while copying
            self.overruns += 1
        struct.pack_into(HDR, _frame, 0, MAGIC, self.seq, self.sent, FRAME,
                         flags)
        crc = _crc16(_frame, 2, HDR_LEN + PAYLOAD, _crc_tab)
        _frame[HDR_LEN + PAYLOAD] = crc & 0xFF
        _frame[HDR_LEN + PAYLOAD + 1] = crc >> 8
        self.sent += FRAME
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self._flags = 0
        return _frame


def _info():
    return ('{"version":"%s","fs_hz":%d,"adc_bits":12,"channels":2,'
            '"pins":[%d,%d],"adc_inputs":[0,1],"frame_samples":%d,'
            '"ring_samples":%d,"magic":"A5E2","crc":"CRC-16/CCITT-FALSE",'
            '"sample_order":"ch0 then ch1, 25 us apart",'
            '"adc_clock_hz":%d,"adc_div_int":%d}'
            % (VERSION, FS_HZ, PINS[0], PINS[1], FRAME, RING_PAIRS,
               ADC_CLOCK_HZ, ADC_CLOCK_HZ // (FS_HZ * 2) - 1))


def main():
    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)
    out = sys.stdout.buffer
    line = bytearray()
    s = Sampler()
    print("INFO " + _info())
    try:
        while True:
            while poller.poll(0):
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch != "\n":
                    if ch != "\r" and len(line) < 32:
                        line.extend(ch.encode())
                    continue
                cmd = bytes(line).strip().upper()
                line = bytearray()
                if cmd == b"INFO":
                    print("INFO " + _info())
                elif cmd == b"START":
                    s.start()
                    print("STARTED")
                elif cmd == b"STOP":
                    s.stop()
                    print("STOPPED %d %d %d" % (s.seq, s.sent, s.overruns))
                elif cmd == b"EXIT":
                    return
                elif cmd:
                    print("ERR unknown command " + cmd.decode())
            frame = s.next_frame() if s.running else None
            if frame is None:
                utime.sleep_ms(1)
            else:
                out.write(frame)
    finally:
        s.stop()


main()
