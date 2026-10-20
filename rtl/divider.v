// Unsigned restoring divider: 32-bit numerator / 24-bit denominator, 32 cycles.
// quo and rem are valid from the cycle done pulses until the next start.
`default_nettype none
module divider (
    input  wire        clk,
    input  wire        start,
    input  wire [31:0] num,
    input  wire [23:0] den,
    output reg  [31:0] quo,
    output wire [23:0] rem,
    output reg         done
);
    reg [24:0] r;
    reg [23:0] d;
    reg [5:0]  n;
    reg        busy = 1'b0;

    wire [24:0] rs = {r[23:0], quo[31]};
    wire        ge = rs >= {1'b0, d};

    assign rem = r[23:0];

    always @(posedge clk) begin
        done <= 1'b0;
        if (start) begin
            r    <= 25'd0;
            d    <= den;
            quo  <= num;
            n    <= 6'd32;
            busy <= 1'b1;
        end else if (busy) begin
            r   <= ge ? rs - {1'b0, d} : rs;
            quo <= {quo[30:0], ge};
            n   <= n - 6'd1;
            if (n == 6'd1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
endmodule
