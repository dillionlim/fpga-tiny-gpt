// Drives llm_core through prompt steps then one autoregressive run, comparing
// every logit and sampled token against train/gen_vectors.py.
`timescale 1ns/1ps
`default_nettype none
module tb_core;
    reg clk = 0, rst = 1;
    always #10 clk = ~clk;

    reg [31:0] vec [0:200000];
    reg [31:0] cfg [0:4];
    initial begin
        $readmemh("vectors.hex", vec);
        $readmemh("cfg.hex", cfg);
    end

    reg start = 0, push = 0, stop = 0, reset_ctx = 0, seed_we = 0;
    reg [5:0] tok_in = 0;
    reg [15:0] nextra = 0;
    reg [5:0] lg_raddr = 0;
    wire [23:0] lg_rdata;
    wire busy, full, tok_valid;
    wire [15:0] pos;
    wire [5:0] last_tok, tok_data;
    wire [31:0] last_cycles;

    llm_core dut (
        .clk(clk), .rst(rst), .start(start), .start_tok(tok_in), .start_push(push),
        .start_nextra(nextra), .start_mt(cfg[1][15:0]), .start_ucut(cfg[2][15:0]),
        .stop(stop), .reset_ctx(reset_ctx), .seed_we(seed_we), .seed(cfg[0]),
        .busy(busy), .pos(pos), .full(full), .last_tok(last_tok), .last_cycles(last_cycles),
        .tok_valid(tok_valid), .tok_data(tok_data), .tok_ready(1'b1),
        .lg_raddr(lg_raddr), .lg_rdata(lg_rdata));

    integer errors = 0, step = 0, i;

    task check_step(input [5:0] sampled);
        begin
            if (sampled !== vec[step * 66 + 1][5:0]) begin
                errors = errors + 1;
                $display("step %0d: sampled %0d expected %0d", step, sampled, vec[step * 66 + 1]);
            end
            for (i = 0; i < 64; i = i + 1) begin
                lg_raddr = i;
                #1;
                if (lg_rdata !== vec[step * 66 + 2 + i][23:0]) begin
                    errors = errors + 1;
                    if (errors < 20)
                        $display("step %0d logit %0d: got %0d expected %0d", step, i,
                                 $signed(lg_rdata), $signed(vec[step * 66 + 2 + i][23:0]));
                end
            end
            if (step < 3 || step % 16 == 0)
                $display("step %0d ok-so-far (errors %0d) tok %0d cycles %0d pos %0d",
                         step, errors, sampled, last_cycles, pos);
            step = step + 1;
        end
    endtask

    initial begin
        #1;
        repeat (5) @(posedge clk);
        rst = 0;
        @(posedge clk) seed_we <= 1;
        @(posedge clk) seed_we <= 0;
        // single steps through the prompt
        while (step < cfg[3]) begin
            @(posedge clk);
            tok_in <= vec[step * 66][5:0]; push <= 0; nextra <= 0; start <= 1;
            @(posedge clk) start <= 0;
            @(posedge clk);
            wait (!busy);
            @(posedge clk);
            check_step(last_tok);
        end
        // one autoregressive run for the remaining steps
        @(posedge clk);
        tok_in <= vec[step * 66][5:0]; push <= 1; nextra <= cfg[4] - cfg[3] - 1; start <= 1;
        @(posedge clk) start <= 0;
        while (step < cfg[4]) begin
            @(posedge clk);
            if (tok_valid) check_step(tok_data);
        end
        wait (!busy);
        $display("DONE: %0d steps, %0d errors", step, errors);
        $finish;
    end

    initial begin
        #4000000000;
        $display("TIMEOUT at step %0d", step);
        $finish;
    end
endmodule
