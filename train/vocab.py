"""Character vocabulary shared by training, export and the badge app."""
import numpy as np

EOS = "\x00"
CHARS = (EOS + "\n" + " " + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         + ".,\"'!?-:;")
assert len(CHARS) == 64 and len(set(CHARS)) == 64

NORMALIZE = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-",
             "—": "-", "…": "...", "\xa0": " "}

STOI = {c: i for i, c in enumerate(CHARS)}


def clean(text):
    text = text.replace("<|endoftext|>", EOS)
    for a, b in NORMALIZE.items():
        text = text.replace(a, b)
    return "".join(c for c in text if c in STOI)


def encode(text):
    return np.array([STOI[c] for c in clean(text)], dtype=np.uint8)


def decode(ids):
    return "".join(CHARS[i] for i in ids)
