// floor(sqrt(x)) for a 40-bit x, one result bit per cycle (20 cycles).
`default_nettype none
module isqrt (
    input  wire        clk,
    input  wire        start,
    input  wire [39:0] x,
    output wire [19:0] root,
    output reg         done
);
    reg [39:0] num, res, bitv;
    reg        busy = 1'b0;

    wire [39:0] trial = res + bitv;

    assign root = res[19:0];

    always @(posedge clk) begin
        done <= 1'b0;
        if (start) begin
            num  <= x;
            res  <= 40'd0;
            bitv <= 40'd1 << 38;
            busy <= 1'b1;
        end else if (busy) begin
            if (num >= trial) begin
                num <= num - trial;
                res <= (res >> 1) + bitv;
            end else begin
                res <= res >> 1;
            end
            bitv <= bitv >> 2;
            if (bitv == 40'd1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
endmodule
