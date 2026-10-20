"""Tiny serial helpers for scripted badge access (no pyserial needed)."""
import glob
import os
import select
import termios
import time


def find_badge():
    for tty in sorted(glob.glob("/sys/class/tty/ttyACM*")):
        try:
            with open(os.path.join(tty, "device/../idVendor")) as f:
                if f.read().strip() == "1337":
                    return "/dev/" + os.path.basename(tty)
        except OSError:
            pass
    return None


def open_badge(dev=None):
    dev = dev or find_badge()
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    a = termios.tcgetattr(fd)
    a[0] = 0; a[1] = 0; a[3] = 0
    a[2] |= termios.CLOCAL | termios.CREAD
    termios.tcsetattr(fd, termios.TCSANOW, a)
    return fd


def read_for(fd, sec, until=None):
    buf = b""
    end = time.time() + sec
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                buf += os.read(fd, 4096)
            except (BlockingIOError, OSError):
                break
            if until and until in buf:
                break
    return buf.decode("utf-8", "replace")
