// Simulation stand-in for the ECP5 internal oscillator (310 MHz / DIV).
`timescale 1ns/1ps
module OSCG #(parameter DIV = 8) (output reg OSC);
    initial OSC = 1'b0;
    always #(1000.0 / 310.0 * DIV / 2) OSC = ~OSC;
endmodule
