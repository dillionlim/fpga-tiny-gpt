// SPI mode 0 slave, oversampled by clk. MSB first.
// byte_valid pulses after each received byte with its index since CS fell.
// The byte to send next is loaded with tx_load (normally the cycle after byte_valid).
`default_nettype none
module spi_slave (
    input  wire       clk,
    input  wire       sck_in,
    input  wire       csn_in,
    input  wire       mosi_in,
    output wire       miso,
    output wire       active,
    output reg        byte_valid,
    output reg  [7:0] byte_data,
    output reg  [8:0] byte_idx,
    input  wire       tx_load,
    input  wire [7:0] tx_byte
);
    reg [2:0] sck_s = 3'b000, cs_s = 3'b111;
    reg [1:0] mosi_s = 2'b00;
    always @(posedge clk) begin
        sck_s  <= {sck_s[1:0], sck_in};
        cs_s   <= {cs_s[1:0], csn_in};
        mosi_s <= {mosi_s[0], mosi_in};
    end
    wire sck_rise = sck_s[2:1] == 2'b01;
    assign active = ~cs_s[1];
    wire cs_fall = cs_s[2:1] == 2'b10;

    reg [2:0] bitcnt = 3'd0;
    reg [6:0] rx_sh = 7'd0;
    reg [7:0] tx_sh = 8'hA5;
    reg [8:0] idx = 9'd0;
    assign miso = tx_sh[7];

    always @(posedge clk) begin
        byte_valid <= 1'b0;
        if (cs_fall) begin
            bitcnt <= 3'd0;
            idx    <= 9'd0;
            tx_sh  <= 8'hA5;
        end else if (active && sck_rise) begin
            rx_sh  <= {rx_sh[5:0], mosi_s[1]};
            tx_sh  <= {tx_sh[6:0], 1'b0};
            bitcnt <= bitcnt + 3'd1;
            if (bitcnt == 3'd7) begin
                byte_valid <= 1'b1;
                byte_data  <= {rx_sh, mosi_s[1]};
                byte_idx   <= idx;
                idx        <= idx + 9'd1;
            end
        end else if (tx_load) begin
            tx_sh <= tx_byte;
        end
    end
endmodule
