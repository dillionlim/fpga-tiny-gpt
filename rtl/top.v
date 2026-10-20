// GreyMecha/Army badge: tiny GPT on the ECP5, driven by the RP2350 over SPI.
//   interconnect[0] GP8  MISO     interconnect[2] GP10 SCK
//   interconnect[1] GP9  CS_N     interconnect[3] GP11 MOSI
// Commands (first byte of a CS-low transaction):
//   0x01 reset context          0x04 stop autoregressive run
//   0x02 run: tok, flags(bit0 push), n_extra[15:8], n_extra[7:0], mt[15:8], mt[7:0], ucut[15:8], ucut[7:0]
//   0x03 seed: 4 bytes, big-endian
//   0x10 status: flags, fifo count, pos[15:8], pos[7:0], last tok, cycles[31:0], 0x4C
//        flags = {btn pressed[4:0], full, fifo nonempty, busy}
//   0x11 read fifo: count, then that many tokens (0xFF after)
//   0x12 read logits: 64 x int32 big-endian
`default_nettype none
module top (
    input  wire [4:0] btn,
    output wire [7:0] led,
    output wire       spi_miso,
    input  wire       spi_csn,
    input  wire       spi_sck,
    input  wire       spi_mosi
);
    wire clk;
    OSCG #(.DIV(`OSC_DIV)) osc (.OSC(clk));

    reg [7:0] por = 8'd0;
    wire rst = ~&por;
    always @(posedge clk) if (rst) por <= por + 8'd1;

    // ---------------------------------------------------------------- SPI
    wire       spi_active, byte_valid;
    wire [7:0] byte_data;
    wire [8:0] byte_idx;
    reg        tx_load = 1'b0;
    reg  [7:0] tx_byte = 8'h00;
    spi_slave u_spi (.clk(clk), .sck_in(spi_sck), .csn_in(spi_csn), .mosi_in(spi_mosi),
                     .miso(spi_miso), .active(spi_active), .byte_valid(byte_valid),
                     .byte_data(byte_data), .byte_idx(byte_idx), .tx_load(tx_load), .tx_byte(tx_byte));

    // ---------------------------------------------------------------- core
    reg         core_start = 1'b0, core_stop = 1'b0, core_reset = 1'b0, seed_we = 1'b0;
    reg  [5:0]  run_tok;
    reg         run_push;
    reg  [15:0] run_nextra, run_mt, run_ucut;
    reg  [31:0] seed;
    wire        busy, full, tok_valid;
    wire [15:0] pos;
    wire [5:0]  last_tok, tok_data;
    wire [31:0] last_cycles;
    wire [5:0]  lg_raddr;
    wire [23:0] lg_rdata;

    // ---------------------------------------------------------------- token fifo
    reg [5:0] fifo [0:255];
    reg [7:0] wr_ptr = 8'd0, rd_ptr = 8'd0;
    reg [8:0] count = 9'd0;
    reg       fifo_pop = 1'b0, fifo_clear = 1'b0;
    wire      fifo_push = tok_valid;
    always @(posedge clk) begin
        if (fifo_push) fifo[wr_ptr] <= tok_data;
        if (fifo_clear || rst) begin
            wr_ptr <= 8'd0;
            rd_ptr <= 8'd0;
            count  <= 9'd0;
        end else begin
            if (fifo_push) wr_ptr <= wr_ptr + 8'd1;
            if (fifo_pop) rd_ptr <= rd_ptr + 8'd1;
            count <= count + fifo_push - fifo_pop;
        end
    end
    wire [5:0] fifo_head = fifo[rd_ptr];

    llm_core u_core (
        .clk(clk), .rst(rst), .start(core_start), .start_tok(run_tok), .start_push(run_push),
        .start_nextra(run_nextra), .start_mt(run_mt), .start_ucut(run_ucut),
        .stop(core_stop), .reset_ctx(core_reset), .seed_we(seed_we), .seed(seed),
        .busy(busy), .pos(pos), .full(full), .last_tok(last_tok), .last_cycles(last_cycles),
        .tok_valid(tok_valid), .tok_data(tok_data), .tok_ready(count < 9'd250),
        .lg_raddr(lg_raddr), .lg_rdata(lg_rdata));

    // ---------------------------------------------------------------- commands
    reg [7:0] cmd = 8'h00;
    reg [7:0] args [0:6];
    reg [7:0] snap = 8'd0;   // fifo count reported at the start of a read
    wire [7:0] c = (byte_idx == 9'd0) ? byte_data : cmd;
    wire [7:0] ri = byte_idx[7:0];  // index of the response byte to load
    assign lg_raddr = ri[7:2];
    wire [31:0] lg32 = {{8{lg_rdata[23]}}, lg_rdata};
    wire [7:0] status = {~btn, full, count != 9'd0, busy};
    wire [7:0] fifo_count8 = (count > 9'd255) ? 8'd255 : count[7:0];

    reg [7:0] resp;
    always @* begin
        resp = 8'h00;
        case (c)
            8'h10: case (ri)
                8'd0: resp = status;
                8'd1: resp = fifo_count8;
                8'd2: resp = pos[15:8];
                8'd3: resp = pos[7:0];
                8'd4: resp = {2'b00, last_tok};
                8'd5: resp = last_cycles[31:24];
                8'd6: resp = last_cycles[23:16];
                8'd7: resp = last_cycles[15:8];
                8'd8: resp = last_cycles[7:0];
                8'd9: resp = 8'h4C;
                default: resp = 8'h00;
            endcase
            8'h11: resp = (ri == 8'd0) ? fifo_count8 : (ri <= snap) ? {2'b00, fifo_head} : 8'hFF;
            8'h12: case (ri[1:0])
                2'd0: resp = lg32[31:24];
                2'd1: resp = lg32[23:16];
                2'd2: resp = lg32[15:8];
                default: resp = lg32[7:0];
            endcase
            default: resp = 8'h00;
        endcase
    end

    always @(posedge clk) begin
        tx_load    <= 1'b0;
        core_start <= 1'b0;
        core_stop  <= 1'b0;
        core_reset <= 1'b0;
        seed_we    <= 1'b0;
        fifo_pop   <= 1'b0;
        fifo_clear <= 1'b0;
        if (byte_valid) begin
            tx_load <= 1'b1;
            tx_byte <= resp;
            if (byte_idx == 9'd0) begin
                cmd <= byte_data;
                if (byte_data == 8'h11) snap <= fifo_count8;
                if (byte_data == 8'h01 && !busy) begin
                    core_reset <= 1'b1;
                    fifo_clear <= 1'b1;
                end
                if (byte_data == 8'h04) core_stop <= 1'b1;
            end else begin
                if (byte_idx <= 9'd7) args[byte_idx - 9'd1] <= byte_data;
                if (cmd == 8'h11 && ri <= snap && ri != 8'd0) fifo_pop <= 1'b1;
                if (cmd == 8'h03 && byte_idx == 9'd4) begin
                    seed    <= {args[0], args[1], args[2], byte_data};
                    seed_we <= 1'b1;
                end
                if (cmd == 8'h02 && byte_idx == 9'd8 && !busy) begin
                    run_tok    <= args[0][5:0];
                    run_push   <= args[1][0];
                    run_nextra <= {args[2], args[3]};
                    run_mt     <= {args[4], args[5]};
                    run_ucut   <= {args[6], byte_data};
                    core_start <= 1'b1;
                end
            end
        end
    end

    assign led = {busy, count != 9'd0, last_tok};
endmodule
