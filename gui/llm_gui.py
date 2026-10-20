"""Desktop window for the badge's FPGA LLM.

  python3 -I llm_gui.py

Finds the badge on USB (VID 1337), asks it to open the Tiny LLM app, then sends
prompts and streams the story back as the FPGA generates it.
"""
import glob
import os
import queue
import re
import select
import termios
import threading
import time
import tkinter as tk
from tkinter import ttk

BG, FG, DIM, ACCENT, PROMPT = "#15161a", "#e8e6e3", "#8a8f98", "#f5c542", "#7cc4ff"
# CircuitPython sends terminal-title (OSC) and cursor (CSI) escapes with no newline after them
ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;]*[A-Za-z]")


def find_badge():
    for tty in sorted(glob.glob("/sys/class/tty/ttyACM*")):
        try:
            with open(os.path.join(tty, "device/../idVendor")) as f:
                if f.read().strip() == "1337":
                    return "/dev/" + os.path.basename(tty)
        except OSError:
            pass
    return None


class Badge(threading.Thread):
    """Owns the serial port; puts ("line", text) / ("state", text) on a queue."""

    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events
        self.fd = None
        self.lock = threading.Lock()

    def send(self, line):
        with self.lock:
            if self.fd is None:
                return False
            try:
                os.write(self.fd, (line + "\r\n").encode("latin-1", "replace"))
                return True
            except OSError:
                return False

    def run(self):
        buf = b""
        while True:
            if self.fd is None:
                dev = find_badge()
                if dev is None:
                    self.events.put(("state", "Badge not found. Plug it in."))
                    time.sleep(1.5)
                    continue
                try:
                    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                except PermissionError:
                    self.events.put(("state", f"No permission for {dev}"))
                    time.sleep(2)
                    continue
                a = termios.tcgetattr(fd)
                a[0] = 0; a[1] = 0; a[3] = 0
                a[2] |= termios.CLOCAL | termios.CREAD
                termios.tcsetattr(fd, termios.TCSANOW, a)
                with self.lock:
                    self.fd = fd
                self.events.put(("state", f"Connected on {dev}, opening Tiny LLM..."))
                self.send("")
                self.send("@llm")
                buf = b""
            try:
                r, _, _ = select.select([self.fd], [], [], 0.2)
                if r:
                    data = os.read(self.fd, 4096)
                    if not data:
                        raise OSError("closed")
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = ESCAPES.sub("", line.decode("utf-8", "replace")).rstrip("\r")
                        self.events.put(("line", text))
            except OSError:
                with self.lock:
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
                    self.fd = None
                self.events.put(("state", "Disconnected, retrying..."))
                time.sleep(1)


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append("\n" if s[i + 1] == "n" else s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.badge = Badge(self.events)
        self.ready = False
        self.left = False  # badge user pressed B: stop pulling it back into the app
        self.generating = False
        self.first_chunk = False

        root.title("Tiny LLM on FPGA")
        root.configure(bg=BG)
        root.geometry("760x560")
        root.minsize(520, 380)

        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(head, text="Tiny LLM", font=("Sans", 16, "bold"), bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(head, text="  84K-param GPT running on the badge's ECP5", font=("Sans", 10),
                 bg=BG, fg=DIM).pack(side="left", pady=(4, 0))

        self.text = tk.Text(root, wrap="word", bg="#1d1f24", fg=FG, insertbackground=FG, relief="flat",
                            font=("Monospace", 12), padx=14, pady=12, highlightthickness=0, height=10)
        self.text.pack(fill="both", expand=True, padx=16, pady=6)
        self.text.tag_configure("prompt", foreground=PROMPT)
        self.text.tag_configure("note", foreground=DIM, font=("Sans", 9))
        self.text.configure(state="disabled")

        ctl = tk.Frame(root, bg=BG)
        ctl.pack(fill="x", padx=16)
        tk.Label(ctl, text="Temperature", bg=BG, fg=DIM).pack(side="left")
        self.temp = tk.DoubleVar(value=0.8)
        tk.Scale(ctl, from_=0.2, to=1.5, resolution=0.05, orient="horizontal", variable=self.temp,
                 bg=BG, fg=FG, highlightthickness=0, troughcolor="#2a2d33", length=140).pack(side="left")
        tk.Label(ctl, text="   Length", bg=BG, fg=DIM).pack(side="left")
        self.length = tk.IntVar(value=400)
        tk.Scale(ctl, from_=50, to=1500, resolution=50, orient="horizontal", variable=self.length,
                 bg=BG, fg=FG, highlightthickness=0, troughcolor="#2a2d33", length=140).pack(side="left")

        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", padx=16, pady=(6, 4))
        self.entry = tk.Entry(row, font=("Sans", 12), bg="#24272d", fg=FG, insertbackground=FG,
                              relief="flat", highlightthickness=1, highlightcolor=ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.insert(0, "Once upon a time")
        self.entry.bind("<Return>", lambda e: self.generate())
        self.go = ttk.Button(row, text="Generate", command=self.generate)
        self.go.pack(side="left", padx=(8, 0))
        self.stop = ttk.Button(row, text="Stop", command=lambda: self.badge.send("@stop"))
        self.stop.pack(side="left", padx=(6, 0))

        self.status = tk.Label(root, text="Starting...", bg=BG, fg=DIM, anchor="w")
        self.status.pack(fill="x", padx=16, pady=(0, 10))

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.badge.start()
        self.root.after(30, self.poll)
        self.root.after(3000, self.nudge)
        self.entry.focus_set()

    def nudge(self):
        if not self.ready and not self.left:
            self.badge.send("@llm")
        self.root.after(3000, self.nudge)

    def append(self, s, tag=None):
        self.text.configure(state="normal")
        self.text.insert("end", s, tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def generate(self):
        if not self.ready or self.generating:
            return
        prompt = self.entry.get().replace("\n", " ")
        if self.text.index("end-1c") != "1.0":
            self.append("\n\n")
        self.generating = True
        self.first_chunk = True
        self.status.configure(text="Generating on the FPGA...")
        self.badge.send(f"@gen {self.temp.get():.2f} {self.length.get()} {prompt}")

    def handle(self, line):
        if line == "@ready":
            self.ready = True
            self.status.configure(text="Ready. Type the start of a story and press Enter.")
        elif line == "@loading":
            self.status.configure(text="Loading the LLM bitstream onto the FPGA...")
        elif line.startswith("@t "):
            chunk = unescape(line[3:])
            self.append(chunk, "prompt" if self.first_chunk else None)
            self.first_chunk = False
        elif line.startswith("@end"):
            parts = line.split()
            self.generating = False
            if len(parts) >= 3:
                n, ms = int(parts[1]), float(parts[2])
                tps = 1000 / ms if ms > 0 else 0
                self.append(f"\n[{n} chars, FPGA {ms:.1f} ms/char = {tps:.0f} chars/s]", "note")
            self.status.configure(text="Ready.")
        elif line.startswith("@error"):
            self.generating = False
            self.status.configure(text="Badge error: " + line[7:])
        elif line == "@bye":
            self.ready = False
            self.left = True
            self.status.configure(text="Badge left the LLM app. Reopen this window to start it again.")

    def poll(self):
        try:
            while True:
                kind, val = self.events.get_nowait()
                if kind == "state":
                    self.ready = False
                    self.status.configure(text=val)
                else:
                    self.handle(val)
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def close(self):
        self.badge.send("@quit")
        self.root.after(150, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
