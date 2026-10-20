#!/bin/bash
# Synthesize, place and route the badge bitstream. Run after gen_rtl.py.
set -euo pipefail
export PATH=/home/dillion/tools/oss-cad-suite/bin:$PATH
RTL=$(cd "$(dirname "$0")" && pwd)
DIV=${OSC_DIV:-8}
GEN=${GEN:-gen}
OUT=${OUT:-llm.bit}
cd "$RTL/$GEN"   # $readmemh paths are relative to here
yosys -q -l "$RTL/build/yosys.log" -p "read_verilog -DOSC_DIV=$DIV -I. $RTL/top.v $RTL/spi_slave.v $RTL/llm_core.v $RTL/divider.v $RTL/isqrt.v; synth_ecp5 -top top -json $RTL/build/top.json"
nextpnr-ecp5 --25k --package CABGA256 --speed 6 --json "$RTL/build/top.json" --lpf "$RTL/pinout.lpf" \
    --textcfg "$RTL/build/top.config" --freq 50 --timing-allow-fail -l "$RTL/build/nextpnr.log" -q
ecppack "$RTL/build/top.config" "$RTL/build/$OUT"
ls -la "$RTL/build/$OUT"
