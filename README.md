# fpga-tiny-gpt

A tiny character-level GPT that runs entirely inside the Lattice ECP5 FPGA on the
[GreyMecha/Army badge](https://github.com/NUSGreyhats/greybadge25). The badge's RP2350 only
loads the bitstream and passes text back and forth over SPI. Every multiply, attention
score and sampled character is computed in FPGA fabric.

| | |
|---|---|
| Model | 3 layers, d=64, 4 heads (multi-query attention), FFN 128, context 128, ~84K params |
| Vocab | 64 characters (`a-z A-Z . , " ' ! ? - : ;`, space, newline, EOS) |
| Training data | TinyStories |
| FPGA | LFE5U-25F, 31 MHz internal oscillator |
| Resources | 48/56 block RAM, ~21% logic, 3/28 DSP |
| Speed | ~5.8 ms per character |

See [`docs/circuit.html`](docs/circuit.html) for wiring diagrams, the on-chip datapath
and the pin map.

## How it works

- **Training** (`train/`): a float model is trained, activation scales are calibrated,
  then it is fine-tuned with quantization-aware training that mimics the hardware
  exactly. That means int8 weights with per-row scales, an int16 Q7.8 residual stream,
  int8 RMSNorm output, attention logits rounded to 1/32 nat and ALiBi position bias.
- **Integer reference** (`train/intref.py`): a bit-exact Python model of the FPGA
  datapath. Every operation has a one-to-one counterpart in the RTL.
- **RTL** (`rtl/`): `llm_core.v` is a sequencer that runs a fixed 32-op program
  (embed, norm, matvec, attention, sample) for each token. All work goes through one
  shared multiply-accumulate unit, with weights in block RAM, an int8 K/V ring buffer,
  an exp lookup table, a divider and an integer square root. `top.v` adds the SPI slave,
  a token FIFO and LEDs. Sampled tokens feed straight back in for autoregressive
  generation.
- **Badge app** (`badge/apps/llm.py`): a CircuitPython app that loads the bitstream
  over JTAG, sends prompts over SPI, shows the story on the round LCD and talks to a
  host over the USB serial console.
- **Desktop GUI** (`gui/llm_gui.py`): a Tkinter window that finds the badge on USB,
  sends prompts and streams the generated text back.

## Repository layout

```
train/      model, training, QAT, export, integer reference, RTL/vector generators
            ckpt_C*.pt (checkpoints), params_C*.pkl (exported integer params)
rtl/        Verilog sources, pinout.lpf, build.sh
rtl/gen*    generated ROM images, op program and test vectors (gen_full = shipped model)
rtl/sim/    testbenches for the core and for the SPI top level
rtl/build/  prebuilt bitstreams (llm.bit is what the badge app loads)
badge/      CircuitPython app for the badge
gui/        desktop GUI and serial helpers
docs/       circuit and datapath write-up
```

## Usage

### Run on the badge

Copy `rtl/build/llm.bit` to `/hardware/bitstreams/llm.bit` and `badge/apps/llm.py` to
`/apps/llm.py` on the badge's CIRCUITPY drive, then open **Tiny LLM** from the menu.
To drive it from a PC:

```sh
python3 -I gui/llm_gui.py
```

Over the serial console the app accepts `@llm`, `@gen <temp> <n> <prompt>`, `@stop` and
`@quit`. The header of `badge/apps/llm.py` describes the protocol.

### Train and export

Training expects TinyStories encoded with `train/vocab.py` as `uint8` arrays at
`data/ts_train_200M.npy` and `data/ts_valid.npy`. The dataset is not included in this repo.
Update the `DATA` path in `train/train.py` and the path in `train/evalint.py` to match
your setup.

```sh
cd train
python3 -I train.py --arch C --steps 20000 --qat-steps 6000 --out ckpt_C.pt
python3 -I export.py ckpt_C.pt params_C.pkl      # integer params
python3 -I evalint.py params_C.pkl               # int model loss + sample
python3 -I gen_rtl.py params_C.pkl ../rtl/gen    # ROMs + op program
python3 -I gen_vectors.py params_C.pkl ../rtl/gen 200
```

### Simulate

Install [Icarus Verilog](https://github.com/steveicarus/iverilog). The testbench
compares every logit and sampled token against the Python reference:

```sh
cd rtl/gen
iverilog -g2012 -I. -o tb ../sim/tb_core.v ../llm_core.v ../divider.v ../isqrt.v
vvp -n tb
```

### Build the bitstream

You need yosys, nextpnr-ecp5 and ecppack, for example from
[oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build). Edit the `PATH` line in
`rtl/build.sh` to point at your install, then run:

```sh
OSC_DIV=10 GEN=gen_full OUT=llm.bit rtl/build.sh   # 310 MHz / 10 = 31 MHz
```

## Credits

The badge hardware and pin assignments come from
[NUSGreyhats/greybadge25](https://github.com/NUSGreyhats/greybadge25).
