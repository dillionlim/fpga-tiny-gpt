### Tiny LLM ###############################################
# A 3-layer character-level GPT running entirely on the ECP5. The RP2350 only
# loads the bitstream, sends the prompt over SPI and shows the story.
#
# Serial protocol (USB console), one command per line:
#   @llm                      -> "@ready"
#   @gen <temp> <n> <prompt>  -> "@t <chunk>" lines, then "@end <tokens> <ms per token>"
#   @stop                     stop generating
#   @quit                     restore the original bitstream and go back to the menu
# Chunks escape "\" as "\\" and newline as "\n".
import board
import busio
import digitalio
import displayio
import os
import supervisor
import sys
import terminalio
import time
from adafruit_display_text import label

import hardware.fpga

BITSTREAM = "/hardware/bitstreams/llm.bit"
CHARS = "\x00\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,\"'!?-:;"
LOGIT_K = 0.0010166363790631294
UCUT = 128            # drop tokens below exp(-128/32) = 1.8% of the top probability
FPGA_HZ = 31_000_000  # OSCG 310 MHz / 10, nominal
STOCK_PROMPTS = ["Once upon a time", "One day, a little cat", "Tom and Sue went to the park",
                 "Lily found a big red box"]

COLS = 26
ROWS = 11


def encode(text):
    out = []
    for ch in text:
        i = CHARS.find(ch)
        if i > 0:
            out.append(i)
    return out


def temp_mult(temp):
    return max(0, min(65535, int(LOGIT_K * 32 / max(temp, 0.05) * 65536)))


class Fpga:
    def __init__(self):
        self.spi = busio.SPI(clock=board.GP10, MOSI=board.GP11, MISO=board.GP8)
        while not self.spi.try_lock():
            pass
        self.spi.configure(baudrate=2_000_000, polarity=0, phase=0)
        self.cs = digitalio.DigitalInOut(board.GP9)
        self.cs.direction = digitalio.Direction.OUTPUT
        self.cs.value = True
        self.inbuf = bytearray(40)

    def deinit(self):
        self.spi.unlock()
        self.spi.deinit()
        self.cs.deinit()

    def xfer(self, out, n_in=0):
        buf = bytearray(out) + bytes(n_in)
        res = self.inbuf if len(buf) <= len(self.inbuf) else bytearray(len(buf))
        res = memoryview(res)[:len(buf)]
        self.cs.value = False
        self.spi.write_readinto(buf, res)
        self.cs.value = True
        return bytes(res)

    def status(self):
        r = self.xfer(b"\x10", 10)
        return {"busy": r[1] & 1, "count": r[2], "pos": (r[3] << 8) | r[4], "last": r[5],
                "cycles": (r[6] << 24) | (r[7] << 16) | (r[8] << 8) | r[9], "magic": r[10]}

    def reset(self):
        self.xfer(b"\x01")

    def stop(self):
        self.xfer(b"\x04")

    def seed(self, s):
        self.xfer(bytes([0x03, (s >> 24) & 255, (s >> 16) & 255, (s >> 8) & 255, s & 255]))

    def run(self, tok, push, nextra, mt, ucut):
        self.xfer(bytes([0x02, tok, 1 if push else 0, nextra >> 8, nextra & 255,
                         mt >> 8, mt & 255, ucut >> 8, ucut & 255]))

    def read_fifo(self):
        r = self.xfer(b"\x11", 33)
        n = min(r[1], 32)
        return r[2:2 + n]

    def wait_idle(self, timeout=2.0):
        end = time.monotonic() + timeout
        while self.status()["busy"]:
            if time.monotonic() > end:
                return False
        return True


class Screen:
    def __init__(self, hw_state):
        self.display = hw_state["display"]
        self.main = hw_state["display"].root_group
        self.main.pop()
        g = displayio.Group()
        self.header = label.Label(terminalio.FONT, text="Tiny LLM on FPGA", color=0xFFFF00,
                                  anchor_point=(0.5, 0.5), anchored_position=(120, 34))
        self.body = label.Label(terminalio.FONT, text="", color=0xFFFFFF, x=42, y=58, line_spacing=1.0)
        self.footer = label.Label(terminalio.FONT, text="", color=0x00FFFF,
                                  anchor_point=(0.5, 0.5), anchored_position=(120, 206))
        g.append(self.header)
        g.append(self.body)
        g.append(self.footer)
        self.main.append(g)
        self.group = g
        self.lines = [""]

    def diag(self, tag):
        d = self.display
        try:
            t = time.monotonic()
            ok = d.refresh()
            dt = time.monotonic() - t
        except Exception as e:
            ok, dt = repr(e), 0
        print("@diag %s root_is_display_root=%s root_len=%d ours_last=%s auto_refresh=%s refresh=%s %.3fs" % (
            tag, d.root_group is self.main, len(self.main), self.main[-1] is self.group,
            d.auto_refresh, ok, dt))

    def clear(self):
        self.lines = [""]
        self.body.text = ""

    def add(self, text):
        for ch in text:
            if ch == "\n":
                self.lines.append("")
                continue
            cur = self.lines[-1] + ch
            if len(cur) > COLS:
                sp = cur.rfind(" ")
                if sp > 0:
                    self.lines[-1] = cur[:sp]
                    cur = cur[sp + 1:]
                else:
                    self.lines[-1] = cur[:-1]
                    cur = ch
                self.lines.append(cur)
            else:
                self.lines[-1] = cur
        self.lines = self.lines[-ROWS:]

    def show(self, footer=None):
        self.body.text = "\n".join(self.lines)
        if footer is not None:
            self.footer.text = footer


def esc(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\x00", "")


def read_line(buf):
    """Non-blocking: returns a full line from the USB console, or None."""
    while supervisor.runtime.serial_bytes_available:
        ch = sys.stdin.read(1)
        if ch in "\r\n":
            if buf:
                line = "".join(buf)
                buf.clear()
                return line
        else:
            buf.append(ch)
    return None


def generate(fpga, screen, prompt, n_tokens, temp, inbuf):
    toks = [0, 1] + encode(prompt)  # end-of-story, newline, prompt
    mt = temp_mult(temp)
    fpga.wait_idle()
    fpga.reset()
    fpga.seed(int.from_bytes(os.urandom(4), "little") | 1)
    screen.clear()
    shown = "".join(CHARS[t] for t in toks[2:])
    screen.add(shown)
    screen.show("thinking...")
    print("@t " + esc(shown))
    for t in toks[:-1]:
        fpga.run(t, False, 0, mt, UCUT)
        fpga.wait_idle()
    fpga.run(toks[-1], True, n_tokens - 1, mt, UCUT)
    got = 0
    stopped = False
    t0 = time.monotonic()
    last_draw = 0
    while True:
        chunk = fpga.read_fifo()
        if chunk:
            text = ""
            for t in chunk:
                if t == 0:
                    stopped = True
                    break
                text += CHARS[t]
            got += len(text)
            if text:
                screen.add(text)
                print("@t " + esc(text))
            if stopped:
                fpga.stop()
        line = read_line(inbuf)
        if line == "@stop":
            fpga.stop()
            stopped = True
        now = time.monotonic()
        if now - last_draw > 0.15:
            screen.show("%d chars" % got)
            last_draw = now
        if not chunk:
            st = fpga.status()
            if not st["busy"] and st["count"] == 0:
                break
    st = fpga.status()
    ms = st["cycles"] * 1000 / FPGA_HZ
    screen.show("%d chars, %.1f ms/char" % (got, ms))
    print("@end %d %.2f" % (got, ms))


def llm_app(hw_state):
    overlay = hw_state["fpga_overlay"]
    screen = Screen(hw_state)
    screen.footer.text = "loading FPGA..."
    print("@loading")
    screen.diag("after-screen")

    # free the pins the default overlay holds, then load our bitstream
    if overlay.existing_state is not None and isinstance(overlay.existing_state, list):
        overlay.deinit_mode_buttons()
    overlay.jtag_rst.deinit()
    jtag_rst = hardware.fpga.upload_bitstream(BITSTREAM)
    fpga = Fpga()
    screen.diag("after-upload")
    st = fpga.status()
    if st["magic"] != 0x4C:
        screen.footer.text = "FPGA not responding"
        print("@error fpga magic 0x%02x" % st["magic"])
    else:
        screen.footer.text = "A: story  B: exit"
        print("@ready")

    inbuf = []
    btn_a, btn_b = hw_state["btn_action"]
    stock = 0
    while True:
        line = read_line(inbuf)
        if line is not None:
            if line == "@llm":
                print("@ready")
            elif line == "@diag":
                screen.diag("cmd")
            elif line == "@quit":
                break
            elif line.startswith("@gen "):
                parts = line.split(" ", 3)
                try:
                    temp, n = float(parts[1]), int(parts[2])
                except (ValueError, IndexError):
                    print("@error bad @gen")
                    continue
                prompt = parts[3] if len(parts) > 3 else ""
                generate(fpga, screen, prompt, max(1, min(n, 2000)), temp, inbuf)
        if not btn_a.value:
            generate(fpga, screen, STOCK_PROMPTS[stock % len(STOCK_PROMPTS)], 400, 0.8, inbuf)
            stock += 1
        if not btn_b.value:
            break
        time.sleep(0.01)

    # hand the FPGA back to the badge firmware
    screen.footer.text = "restoring FPGA..."
    fpga.stop()
    fpga.deinit()
    jtag_rst.deinit()
    overlay.init()
    print("@bye")
    time.sleep(0.3)
