// SPI-level test of top.v: seed, run 5 tokens, poll status, read fifo and logits.
`timescale 1ns/1ps
`default_nettype none
module tb_top;
    reg sck = 0, csn = 1, mosi = 0;
    wire miso;
    wire [7:0] led;
    top dut (.btn(5'b11111), .led(led), .spi_miso(miso), .spi_csn(csn), .spi_sck(sck), .spi_mosi(mosi));

    reg [31:0] vec [0:1000];
    reg [31:0] cfg [0:4];
    initial begin
        $readmemh("vectors.hex", vec);
        $readmemh("cfg.hex", cfg);
    end

    localparam HALF = 250;  // 2 MHz SCK
    reg [7:0] rx [0:300];
    integer errors = 0, i, n;

    task xfer_byte(input [7:0] tx, output [7:0] got);
        integer b;
        begin
            got = 0;
            for (b = 7; b >= 0; b = b - 1) begin
                mosi = tx[b];
                #HALF sck = 1;
                got[b] = miso;
                #HALF sck = 0;
            end
        end
    endtask

    // tx bytes in txb[0..len-1]; responses land in rx[]
    reg [7:0] txb [0:300];
    task xfer(input integer len);
        integer k;
        reg [7:0] g;
        begin
            csn = 0;
            #HALF;
            for (k = 0; k < len; k = k + 1) begin
                xfer_byte(txb[k], g);
                rx[k] = g;
            end
            #HALF csn = 1;
            #(4 * HALF);
        end
    endtask

    task status;
        integer k;
        begin
            txb[0] = 8'h10;
            for (k = 1; k < 11; k = k + 1) txb[k] = 0;
            xfer(11);
        end
    endtask

    initial begin
        #20000;
        $display("%0t first status", $time);
        status;
        $display("status: flags %02x count %0d pos %0d magic %02x", rx[1], rx[2], {rx[3], rx[4]}, rx[10]);
        if (rx[10] !== 8'h4C) begin errors = errors + 1; $display("bad magic"); end

        txb[0] = 8'h03; txb[1] = cfg[0][31:24]; txb[2] = cfg[0][23:16]; txb[3] = cfg[0][15:8]; txb[4] = cfg[0][7:0];
        xfer(5);

        txb[0] = 8'h02; txb[1] = vec[0][5:0]; txb[2] = 8'h01; txb[3] = 8'h00; txb[4] = 8'h04;
        txb[5] = cfg[1][15:8]; txb[6] = cfg[1][7:0]; txb[7] = cfg[2][15:8]; txb[8] = cfg[2][7:0];
        xfer(9);
        status;
        if (rx[1][0] !== 1'b1) begin errors = errors + 1; $display("not busy after run"); end
        n = 0;
        while (rx[1][0] === 1'b1 && n < 60) begin
            #1000000;
            if (n % 5 == 0) $display("%0t poll %0d flags %02x pos %0d", $time, n, rx[1], {rx[3], rx[4]});
            status;
            n = n + 1;
        end
        $display("done after %0d polls: flags %02x count %0d pos %0d last %0d cycles %0d",
                 n, rx[1], rx[2], {rx[3], rx[4]}, rx[5], {rx[6], rx[7], rx[8], rx[9]});

        txb[0] = 8'h11;
        for (i = 1; i < 10; i = i + 1) txb[i] = 0;
        xfer(10);
        $display("fifo: count %0d tokens %0d %0d %0d %0d %0d extra %02x", rx[1], rx[2], rx[3], rx[4], rx[5], rx[6], rx[7]);
        if (rx[1] !== 5) begin errors = errors + 1; $display("bad fifo count"); end
        for (i = 0; i < 5; i = i + 1)
            if (rx[2 + i] !== vec[i * 66 + 1][7:0]) begin
                errors = errors + 1;
                $display("token %0d: got %0d expected %0d", i, rx[2 + i], vec[i * 66 + 1]);
            end
        if (rx[7] !== 8'hFF) begin errors = errors + 1; $display("expected 0xFF past end"); end

        txb[0] = 8'h12;
        for (i = 1; i < 257; i = i + 1) txb[i] = 0;
        xfer(257);
        for (i = 0; i < 64; i = i + 1)
            if ({rx[1 + 4 * i], rx[2 + 4 * i], rx[3 + 4 * i], rx[4 + 4 * i]} !==
                {{8{vec[4 * 66 + 2 + i][23]}}, vec[4 * 66 + 2 + i][23:0]}) begin
                errors = errors + 1;
                if (errors < 10) $display("logit %0d mismatch", i);
            end

        txb[0] = 8'h01;
        xfer(1);
        status;
        if ({rx[3], rx[4]} !== 0) begin errors = errors + 1; $display("pos not reset"); end
        $display("DONE: %0d errors", errors);
        $finish;
    end
endmodule
