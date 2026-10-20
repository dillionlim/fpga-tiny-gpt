// Tiny GPT inference core. Runs one token per start (plus n_extra autoregressive
// tokens), executing the op program in gen/program.vh. Bit-exact with
// train/intref.py.
//
// Memories (all synchronous read, one cycle):
//   wrom   int8 weights          mrom  per-row requant multipliers
//   exprom exp(-u/32) * 65535    kvram K/V ring buffer, {layer, slot, k/v, j}
//   act    int8 activations      xram  int16 residual stream
//   zbuf   attention / sampling scratch      lgram int logits
`default_nettype none
module llm_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [5:0]  start_tok,
    input  wire        start_push,
    input  wire [15:0] start_nextra,
    input  wire [15:0] start_mt,
    input  wire [15:0] start_ucut,
    input  wire        stop,
    input  wire        reset_ctx,
    input  wire        seed_we,
    input  wire [31:0] seed,
    output reg         busy,
    output reg  [15:0] pos,
    output reg         full,
    output reg  [5:0]  last_tok,
    output reg  [31:0] last_cycles,
    output reg         tok_valid = 1'b0,
    output reg  [5:0]  tok_data,
    input  wire        tok_ready,
    input  wire [5:0]  lg_raddr,
    output wire [23:0] lg_rdata
);
`include "params.vh"
    localparam D = 64, HD = 16, NH = 4, V = 64, CTX = 128;

    localparam OP_END = 0, OP_EMBED = 1, OP_NORM = 2, OP_MATVEC = 3, OP_ATTN = 4, OP_SAMPLE = 5;
    localparam M_ACT = 0, M_RELU = 1, M_KC = 2, M_VC = 3, M_RESID = 4, M_LOGIT = 5;

    // ---------------------------------------------------------------- memories
    reg [7:0] wrom [0:W_DEPTH-1];
    initial $readmemh("weights.hex", wrom);
    reg [W_AW-1:0] waddr;
    reg [7:0]      wdata;
    always @(posedge clk) wdata <= wrom[waddr];

    reg [15:0] mrom [0:M_DEPTH-1];
    initial $readmemh("mults.hex", mrom);
    reg [M_AW-1:0] maddr;
    reg [15:0]     mdata;
    always @(posedge clk) mdata <= mrom[maddr];

    reg [15:0] exprom [0:511];
    initial $readmemh("exp.hex", exprom);
    reg [8:0]  eaddr;
    reg [15:0] edata;
    always @(posedge clk) edata <= exprom[eaddr];

    reg [7:0] kvram [0:KV_DEPTH-1];
    reg [KV_AW-1:0] kv_raddr, kv_waddr;
    reg [7:0]       kv_rdata, kv_wdata;
    reg             kv_we;
    always @(posedge clk) begin
        if (kv_we) kvram[kv_waddr] <= kv_wdata;
        kv_rdata <= kvram[kv_raddr];
    end

    reg [7:0] act [0:511];
    reg [8:0] a_raddr, a_waddr;
    reg [7:0] a_rdata, a_wdata;
    reg       a_we;
    always @(posedge clk) begin
        if (a_we) act[a_waddr] <= a_wdata;
        a_rdata <= act[a_raddr];
    end

    reg [15:0] xram [0:63];
    reg [5:0]  x_raddr, x_waddr;
    reg [15:0] x_rdata, x_wdata;
    reg        x_we;
    always @(posedge clk) begin
        if (x_we) xram[x_waddr] <= x_wdata;
        x_rdata <= xram[x_raddr];
    end

    reg [31:0] zbuf [0:127];
    reg [6:0]  z_raddr, z_waddr;
    reg [31:0] z_rdata, z_wdata;
    reg        z_we;
    always @(posedge clk) begin
        if (z_we) zbuf[z_waddr] <= z_wdata;
        z_rdata <= zbuf[z_raddr];
    end

    reg [23:0] lgram [0:63];
    reg [5:0]  lg_waddr, lg_craddr;
    reg [23:0] lg_wdata, lg_crdata;
    reg        lg_we;
    always @(posedge clk) begin
        if (lg_we) lgram[lg_waddr] <= lg_wdata;
        lg_crdata <= lgram[lg_craddr];
    end
    assign lg_rdata = lgram[lg_raddr];

    // ---------------------------------------------------------------- op fields
    reg [3:0]  op_code;
    reg [2:0]  op_mode;
    reg [1:0]  op_layer;
    reg [8:0]  op_in, op_cols, op_rows, op_dst;
    reg [16:0] op_w;
    reg [10:0] op_m;
    reg [5:0]  op_shift;
    reg [15:0] op_attm;
`include "program.vh"

    // ---------------------------------------------------------------- arithmetic units
    reg         div_start;
    reg  [31:0] div_num;
    reg  [23:0] div_den;
    wire [31:0] div_quo;
    wire [23:0] div_rem;
    wire        div_done;
    divider u_div (.clk(clk), .start(div_start), .num(div_num), .den(div_den),
                   .quo(div_quo), .rem(div_rem), .done(div_done));

    reg         sq_start;
    reg  [39:0] sq_x;
    wire [19:0] sq_root;
    wire        sq_done;
    isqrt u_sqrt (.clk(clk), .start(sq_start), .x(sq_x), .root(sq_root), .done(sq_done));

    // requant multiplier: signed 24-bit x unsigned 16-bit
    reg  signed [23:0] rq_a;
    reg         [15:0] rq_b;
    wire signed [41:0] rq_prod = rq_a * $signed({1'b0, rq_b});
    wire signed [41:0] rq_round = rq_prod + (42'sd1 <<< (op_shift - 6'd1));
    wire signed [41:0] rq_out = rq_round >>> op_shift;
    wire signed [31:0] z_samp = {{6{rq_prod[41]}}, rq_prod[41:16]};  // (logit * mt) >> 16

    // ---------------------------------------------------------------- state
    localparam S_IDLE = 0, S_TOK = 1, S_FETCH = 2, S_DISPATCH = 3, S_NEXT = 4,
               S_EMB = 5,
               S_NM_SS = 6, S_NM_SQ = 7, S_NM_SQW = 8, S_NM_RD = 9, S_NM_RD1 = 10, S_NM_RD2 = 11,
               S_NM_DW = 12,
               S_MV_ROW = 13, S_MV_MAC = 14, S_MV_RQ1 = 15, S_MV_RQ2 = 16, S_MV_WR = 17,
               S_AT_HEAD = 18, S_AT_P1 = 19, S_AT_P1R1 = 20, S_AT_P1R2 = 21, S_AT_P1WR = 22,
               S_AT_P2RD = 23, S_AT_P2A = 24, S_AT_P2B = 25, S_AT_P2C = 26, S_AT_P2D = 27,
               S_AT_P3J = 28, S_AT_P3 = 29, S_AT_P3D = 30, S_AT_P3W = 31,
               S_SM_P1RD = 32, S_SM_P1A = 33, S_SM_P1B = 34, S_SM_P1C = 35,
               S_SM_P2RD = 36, S_SM_P2A = 37, S_SM_P2B = 38, S_SM_P2C = 39, S_SM_P2D = 40,
               S_SM_RNG = 41, S_SM_DIV = 42, S_SM_DW = 43,
               S_SM_P3RD = 44, S_SM_P3A = 45, S_SM_P3B = 46,
               S_DONE = 47, S_PUSH = 48, S_AFTER = 49;
    reg [5:0] state;

    reg [5:0]  pc;
    reg [5:0]  tok;
    reg        push;
    reg [15:0] nextra, mt, ucut;
    reg        stop_req;
    reg [31:0] rng;
    reg [31:0] cycles;

    reg [8:0]  c, k;            // issue / consume counters
    reg [8:0]  r;               // matvec row, norm element
    reg [16:0] wp;              // weight pointer
    reg [10:0] mp;              // multiplier pointer
    reg [7:0]  dd, nd;          // attention distance / number of positions
    reg [4:0]  j;
    reg [2:0]  h;
    reg [6:0]  v;               // vocab index in sampling
    reg        v0, v1;          // read pipeline valid flags

    reg signed [39:0] acc;
    reg        [39:0] ss;
    reg        [19:0] s_norm;
    reg               neg;
    reg signed [41:0] t;
    reg signed [31:0] zmax;
    reg        [31:0] sumw, cum, rsel;
    reg               ucut_ok;

    wire [6:0] slot_cur = pos[6:0];
    wire [6:0] slot_c   = pos[6:0] - c[6:0];
    wire [6:0] slot_dd  = pos[6:0] - dd[6:0];

    // shared 17x8 MAC multiplier, operands chosen by state
    reg signed [16:0] mac_a;
    reg signed [7:0]  mac_b;
    always @* begin
        case (state)
            S_AT_P1: begin mac_a = {{9{a_rdata[7]}}, a_rdata}; mac_b = kv_rdata; end
            S_AT_P3: begin mac_a = {1'b0, z_rdata[15:0]};      mac_b = kv_rdata; end
            default: begin mac_a = {{9{wdata[7]}}, wdata};     mac_b = a_rdata;  end
        endcase
    end
    wire signed [24:0] mac_p = mac_a * mac_b;

    wire signed [15:0] xs = x_rdata;
    wire signed [31:0] x_sq = xs * xs;

    function [7:0] clamp8;
        input signed [41:0] val;
        input lo0;  // clamp low end to 0 instead of -127
        begin
            if (val > 42'sd127) clamp8 = 8'd127;
            else if (lo0 && val < 42'sd0) clamp8 = 8'd0;
            else if (val < -42'sd127) clamp8 = 8'h81;
            else clamp8 = val[7:0];
        end
    endfunction

    function [15:0] clamp16;
        input signed [42:0] val;
        begin
            if (val > 43'sd32767) clamp16 = 16'h7fff;
            else if (val < -43'sd32767) clamp16 = 16'h8001;
            else clamp16 = val[15:0];
        end
    endfunction

    function [31:0] xorshift32;
        input [31:0] x;
        reg [31:0] y;
        begin
            y = x ^ (x << 13);
            y = y ^ (y >> 17);
            xorshift32 = y ^ (y << 5);
        end
    endfunction

    // |val| + den/2, the numerator of a round-half-away-from-zero division
    function [31:0] rnum;
        input signed [39:0] val;
        input [23:0] den;
        reg [39:0] mag;
        begin
            mag = val[39] ? -val : val;
            rnum = mag[31:0] + {9'd0, den[23:1]};
        end
    endfunction

    wire signed [32:0] z_minus = $signed({zmax[31], zmax}) - $signed({z_rdata[31], z_rdata});
    wire        [8:0]  u_lut = (z_minus > 33'sd511) ? 9'd511 : z_minus[8:0];
    wire        [9:0]  alibi = ({dd, 6'd0} >> (2 * h)) >> 3;  // (dd * {64,16,4,1}[h]) >> 3

    always @(posedge clk) begin
        div_start <= 1'b0;
        sq_start  <= 1'b0;
        kv_we     <= 1'b0;
        a_we      <= 1'b0;
        x_we      <= 1'b0;
        z_we      <= 1'b0;
        lg_we     <= 1'b0;
        tok_valid <= 1'b0;
        v1        <= v0;
        if (state != S_IDLE) cycles <= cycles + 1;
        if (stop) stop_req <= 1'b1;
        if (seed_we) rng <= seed;

        if (rst) begin
            state    <= S_IDLE;
            busy     <= 1'b0;
            pos      <= 16'd0;
            full     <= 1'b0;
            rng      <= 32'd1;
            v0       <= 1'b0;
            v1       <= 1'b0;
            last_tok <= 6'd0;
            last_cycles <= 32'd0;
        end else case (state)
            S_IDLE: begin
                if (reset_ctx) begin
                    pos  <= 16'd0;
                    full <= 1'b0;
                end
                if (start) begin
                    tok      <= start_tok;
                    push     <= start_push;
                    nextra   <= start_nextra;
                    mt       <= start_mt;
                    ucut     <= start_ucut;
                    stop_req <= 1'b0;
                    busy     <= 1'b1;
                    state    <= S_TOK;
                end
            end

            S_TOK: begin
                pc     <= 6'd0;
                cycles <= 32'd0;
                nd     <= full ? 8'd128 : pos[7:0] + 8'd1;
                state  <= S_FETCH;
            end

            S_FETCH: begin
                load_op(pc);
                state <= S_DISPATCH;
            end

            S_DISPATCH: begin
                c   <= 9'd0;
                k   <= 9'd0;
                r   <= 9'd0;
                acc <= 40'sd0;
                wp  <= op_w;
                mp  <= op_m;
                h   <= 3'd0;
                v   <= 7'd0;
                ss  <= 40'd0;
                zmax <= 32'h80000000;
                sumw <= 32'd0;
                case (op_code)
                    OP_EMBED:  begin wp <= op_w + tok * D; state <= S_EMB; end
                    OP_NORM:   state <= S_NM_SS;
                    OP_MATVEC: state <= S_MV_ROW;
                    OP_ATTN:   state <= S_AT_HEAD;
                    OP_SAMPLE: state <= S_SM_P1RD;
                    default:   state <= S_DONE;
                endcase
            end

            S_NEXT: begin
                pc    <= pc + 6'd1;
                state <= S_FETCH;
            end

            // ------------------------------------------------ EMBED: x = emb[tok] << shift
            S_EMB: begin
                v0 <= 1'b0;
                if (c < D) begin
                    waddr <= wp[W_AW-1:0];
                    wp    <= wp + 17'd1;
                    c     <= c + 9'd1;
                    v0    <= 1'b1;
                end
                if (v1) begin
                    x_we    <= 1'b1;
                    x_waddr <= k[5:0];
                    x_wdata <= $signed(wdata) <<< op_shift;
                    k       <= k + 9'd1;
                    if (k == D - 1) state <= S_NEXT;
                end
            end

            // ------------------------------------------------ NORM: act[dst+i] = x[i] * 2^(g+2) / isqrt(ss/4)
            S_NM_SS: begin
                v0 <= 1'b0;
                if (c < D) begin
                    x_raddr <= c[5:0];
                    c       <= c + 9'd1;
                    v0      <= 1'b1;
                end
                if (v1) begin
                    ss <= ss + {8'd0, x_sq};
                    k  <= k + 9'd1;
                    if (k == D - 1) state <= S_NM_SQ;
                end
            end
            S_NM_SQ: begin
                sq_x     <= ss >> 2;
                sq_start <= 1'b1;
                state    <= S_NM_SQW;
            end
            S_NM_SQW: if (sq_done) begin
                s_norm <= (sq_root == 20'd0) ? 20'd1 : sq_root;
                r      <= 9'd0;
                state  <= S_NM_RD;
            end
            S_NM_RD: begin
                x_raddr <= r[5:0];
                state   <= S_NM_RD1;
            end
            S_NM_RD1: state <= S_NM_RD2;
            S_NM_RD2: begin
                neg       <= xs[15];
                div_num   <= rnum($signed(xs) <<< (NORM_G + 2), {4'd0, s_norm});
                div_den   <= {4'd0, s_norm};
                div_start <= 1'b1;
                state     <= S_NM_DW;
            end
            S_NM_DW: if (div_done) begin
                a_we    <= 1'b1;
                a_waddr <= op_dst + r;
                a_wdata <= clamp8(neg ? -$signed({10'd0, div_quo}) : $signed({10'd0, div_quo}), 1'b0);
                r       <= r + 9'd1;
                state   <= (r == D - 1) ? S_NEXT : S_NM_RD;
            end

            // ------------------------------------------------ MATVEC: one row at a time
            S_MV_ROW: begin
                acc     <= 40'sd0;
                c       <= 9'd0;
                k       <= 9'd0;
                maddr   <= mp[M_AW-1:0];
                x_raddr <= r[5:0];
                state   <= S_MV_MAC;
            end
            S_MV_MAC: begin
                v0 <= 1'b0;
                if (c < op_cols) begin
                    waddr   <= wp[W_AW-1:0];
                    a_raddr <= op_in + c;
                    wp      <= wp + 17'd1;
                    c       <= c + 9'd1;
                    v0      <= 1'b1;
                end
                if (v1) begin
                    acc <= acc + mac_p;
                    k   <= k + 9'd1;
                    if (k == op_cols - 1) state <= S_MV_RQ1;
                end
            end
            S_MV_RQ1: begin
                rq_a  <= acc[23:0];
                rq_b  <= mdata;
                state <= S_MV_RQ2;
            end
            S_MV_RQ2: begin
                t     <= rq_out;
                state <= S_MV_WR;
            end
            S_MV_WR: begin
                case (op_mode)
                    M_ACT, M_RELU: begin
                        a_we    <= 1'b1;
                        a_waddr <= op_dst + r;
                        a_wdata <= clamp8(t, op_mode == M_RELU);
                    end
                    M_KC, M_VC: begin
                        kv_we    <= 1'b1;
                        kv_waddr <= {op_layer[LB-1:0], slot_cur, op_mode == M_VC, r[3:0]};
                        kv_wdata <= clamp8(t, 1'b0);
                    end
                    M_RESID: begin
                        x_we    <= 1'b1;
                        x_waddr <= r[5:0];
                        x_wdata <= clamp16($signed({{27{x_rdata[15]}}, x_rdata}) + t);
                    end
                    default: begin  // M_LOGIT
                        lg_we    <= 1'b1;
                        lg_waddr <= r[5:0];
                        lg_wdata <= acc[23:0];
                    end
                endcase
                r     <= r + 9'd1;
                mp    <= mp + 11'd1;
                state <= (r == op_rows - 1) ? S_NEXT : S_MV_ROW;
            end

            // ------------------------------------------------ ATTN (multi-query, ALiBi)
            S_AT_HEAD: begin
                dd    <= 8'd0;
                c     <= 9'd0;
                k     <= 9'd0;
                acc   <= 40'sd0;
                zmax  <= 32'h80000000;
                sumw  <= 32'd0;
                state <= S_AT_P1;
            end
            // pass 1: z[dd] = requant(q_h . k[slot]) - alibi, track max
            S_AT_P1: begin
                v0 <= 1'b0;
                if (c < HD) begin
                    a_raddr  <= op_in + {h, 4'd0} + c;
                    kv_raddr <= {op_layer[LB-1:0], slot_dd, 1'b0, c[3:0]};
                    c        <= c + 9'd1;
                    v0       <= 1'b1;
                end
                if (v1) begin
                    acc <= acc + mac_p;
                    k   <= k + 9'd1;
                    if (k == HD - 1) state <= S_AT_P1R1;
                end
            end
            S_AT_P1R1: begin
                rq_a  <= acc[23:0];
                rq_b  <= op_attm;
                state <= S_AT_P1R2;
            end
            S_AT_P1R2: begin
                t     <= rq_out - $signed({32'd0, alibi});
                state <= S_AT_P1WR;
            end
            S_AT_P1WR: begin
                z_we    <= 1'b1;
                z_waddr <= dd[6:0];
                z_wdata <= t[31:0];
                if ($signed(t[31:0]) > zmax) zmax <= t[31:0];
                c       <= 9'd0;
                k       <= 9'd0;
                acc     <= 40'sd0;
                if (dd == nd - 1) begin
                    dd    <= 8'd0;
                    state <= S_AT_P2RD;
                end else begin
                    dd    <= dd + 8'd1;
                    state <= S_AT_P1;
                end
            end
            // pass 2: w[dd] = exp_lut[min(zmax - z, 511)], sumw += w
            S_AT_P2RD: begin
                z_raddr <= dd[6:0];
                state   <= S_AT_P2A;
            end
            S_AT_P2A: state <= S_AT_P2B;
            S_AT_P2B: begin
                eaddr <= u_lut;
                state <= S_AT_P2C;
            end
            S_AT_P2C: state <= S_AT_P2D;
            S_AT_P2D: begin
                z_we    <= 1'b1;
                z_waddr <= dd[6:0];
                z_wdata <= {16'd0, edata};
                sumw    <= sumw + {16'd0, edata};
                if (dd == nd - 1) begin
                    j     <= 5'd0;
                    state <= S_AT_P3J;
                end else begin
                    dd    <= dd + 8'd1;
                    state <= S_AT_P2RD;
                end
            end
            // pass 3: out[j] = round(sum w[dd] * v[slot][j] / sumw)
            S_AT_P3J: begin
                acc   <= 40'sd0;
                c     <= 9'd0;
                k     <= 9'd0;
                state <= S_AT_P3;
            end
            S_AT_P3: begin
                v0 <= 1'b0;
                if (c < nd) begin
                    z_raddr  <= c[6:0];
                    kv_raddr <= {op_layer[LB-1:0], slot_c, 1'b1, j[3:0]};
                    c        <= c + 9'd1;
                    v0       <= 1'b1;
                end
                if (v1) begin
                    acc <= acc + mac_p;
                    k   <= k + 9'd1;
                    if (k == nd - 1) state <= S_AT_P3D;
                end
            end
            S_AT_P3D: begin
                neg       <= acc[39];
                div_num   <= rnum(acc, sumw[23:0]);
                div_den   <= sumw[23:0];
                div_start <= 1'b1;
                state     <= S_AT_P3W;
            end
            S_AT_P3W: if (div_done) begin
                a_we    <= 1'b1;
                a_waddr <= op_dst + {h, 4'd0} + j;
                a_wdata <= clamp8(neg ? -$signed({10'd0, div_quo}) : $signed({10'd0, div_quo}), 1'b0);
                if (j == HD - 1) begin
                    if (h == NH - 1) state <= S_NEXT;
                    else begin
                        h     <= h + 3'd1;
                        state <= S_AT_HEAD;
                    end
                end else begin
                    j     <= j + 5'd1;
                    state <= S_AT_P3J;
                end
            end

            // ------------------------------------------------ SAMPLE
            // pass 1: z[v] = (logit[v] * mt) >> 16, track max
            S_SM_P1RD: begin
                lg_craddr <= v[5:0];
                state     <= S_SM_P1A;
            end
            S_SM_P1A: state <= S_SM_P1B;
            S_SM_P1B: begin
                rq_a  <= lg_crdata;
                rq_b  <= mt;
                state <= S_SM_P1C;
            end
            S_SM_P1C: begin
                z_we    <= 1'b1;
                z_waddr <= v;
                z_wdata <= z_samp;
                if (z_samp > zmax) zmax <= z_samp;
                if (v == V - 1) begin
                    v     <= 7'd0;
                    state <= S_SM_P2RD;
                end else begin
                    v     <= v + 7'd1;
                    state <= S_SM_P1RD;
                end
            end
            // pass 2: w[v] = (u <= ucut) ? exp_lut[min(u, 511)] : 0
            S_SM_P2RD: begin
                z_raddr <= v;
                state   <= S_SM_P2A;
            end
            S_SM_P2A: state <= S_SM_P2B;
            S_SM_P2B: begin
                eaddr   <= u_lut;
                ucut_ok <= z_minus <= $signed({17'd0, ucut});
                state   <= S_SM_P2C;
            end
            S_SM_P2C: state <= S_SM_P2D;
            S_SM_P2D: begin
                z_we    <= 1'b1;
                z_waddr <= v;
                z_wdata <= ucut_ok ? {16'd0, edata} : 32'd0;
                sumw    <= sumw + (ucut_ok ? {16'd0, edata} : 32'd0);
                if (v == V - 1) begin
                    v     <= 7'd0;
                    state <= S_SM_RNG;
                end else begin
                    v     <= v + 7'd1;
                    state <= S_SM_P2RD;
                end
            end
            S_SM_RNG: begin
                rng   <= xorshift32(rng);
                state <= S_SM_DIV;
            end
            S_SM_DIV: begin
                div_num   <= rng;
                div_den   <= sumw[23:0];
                div_start <= 1'b1;
                state     <= S_SM_DW;
            end
            S_SM_DW: if (div_done) begin
                rsel  <= {8'd0, div_rem};
                cum   <= 32'd0;
                state <= S_SM_P3RD;
            end
            // pass 3: first v whose cumulative weight exceeds rng % sumw
            S_SM_P3RD: begin
                z_raddr <= v;
                state   <= S_SM_P3A;
            end
            S_SM_P3A: state <= S_SM_P3B;
            S_SM_P3B: begin
                if (cum + z_rdata > rsel || v == V - 1) begin
                    tok   <= v[5:0];
                    state <= S_NEXT;
                end else begin
                    cum   <= cum + z_rdata;
                    v     <= v + 7'd1;
                    state <= S_SM_P3RD;
                end
            end

            // ------------------------------------------------ token finished
            S_DONE: begin
                last_tok    <= tok;
                last_cycles <= cycles;
                pos         <= pos + 16'd1;
                if (pos[6:0] == CTX - 1) full <= 1'b1;
                state       <= push ? S_PUSH : S_AFTER;
            end
            S_PUSH: if (tok_ready) begin
                tok_valid <= 1'b1;
                tok_data  <= tok;
                state     <= S_AFTER;
            end
            S_AFTER: begin
                if (nextra != 16'd0 && !stop_req) begin
                    nextra <= nextra - 16'd1;
                    state  <= S_TOK;
                end else begin
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
        endcase
    end
endmodule
