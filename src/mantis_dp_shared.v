// mantis_dp_shared: T7 organization (b), one time-multiplexed MAC.
// Chosen organization (docs/feasibility.md, 2026-11-23).
//
// One fp8_mul and one two-stage BF16 adder, 1 MAC per clock, round-robin over
// the four accumulators. Each accumulator is read once per 4-clock step
// interval and written 2 clocks after it is read, so there is no
// read-after-write hazard on an accumulator (R10, P11).
//
// Pipeline (issue = 4th WRITE sampled at edge t0), product j = 0..3:
//   edge t0     : op regs <= {slot0, slot1, slot2, din}, run <= 1, cnt <= 0
//   cycle t0+j  : M  - multiply product j (C00, C10, C01, C11)
//   edge t0+j+1 : prod/ptag/pv registered
//   cycle t0+j+1: A1 - bf16_add_s1(acc[ptag], prod): specials, align, add
//   edge t0+j+2 : stage-1 bundle/qtag/qv registered
//   cycle t0+j+2: A2 - bf16_add_s2: normalize, round, pack
//   edge t0+j+3 : acc[qtag] commits; last (j=3) at t0+6 -> drain = 6 (R11)
// The adder is split so that each stage meets 20 ns at ss_100C_1v60.
// A new issue at edge t0+4 reloads the op regs just as product 3 leaves them.
// CLEAR or reset clears run/pv/qv and the accumulators (R13, R14, P10).
`default_nettype none

module mantis_dp_shared (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [1:0]  cmd,
    input  wire [7:0]  din,
    output wire        busy,        // R11, P2
    output wire [63:0] acc,         // {C11, C10, C01, C00}
    output wire        fmt_a,
    output wire        fmt_b,
    output wire [1:0]  n
);
    wire [7:0] a0, a1, b0;
    wire       issue, clear;

    mantis_stage u_stage (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .cmd(cmd),
                          .din(din), .a0(a0), .a1(a1), .b0(b0), .issue(issue),
                          .clear(clear), .fmt_a(fmt_a), .fmt_b(fmt_b), .n(n));

    reg  [7:0]  oa0, oa1, ob0, ob1;   // operand snapshot of the issued step
    reg         run;
    reg  [1:0]  cnt;                  // product j in the multiplier
    reg         pv, qv;               // stage M -> A1, A1 -> A2 valids
    reg  [1:0]  ptag, qtag;           // accumulator index {i, j}
    reg  [15:0] prod;
    reg  [15:0] c [0:3];              // index 2*i + j: C00, C01, C10, C11
    // A1 -> A2 bundle
    reg  [11:0] q_sum;
    reg  [7:0]  q_e_big;
    reg         q_s_big, q_spec;
    reg  [15:0] q_spec_val;

    // product order j = 0..3 -> C00 (A0B0), C10 (A1B0), C01 (A0B1), C11 (A1B1)
    wire [7:0]  mul_a = cnt[0] ? oa1 : oa0;
    wire [7:0]  mul_b = cnt[1] ? ob1 : ob0;
    wire [1:0]  tag   = {cnt[0], cnt[1]};
    wire [15:0] m, s;
    wire [11:0] sum;
    wire [7:0]  e_big;
    wire        s_big, spec;
    wire [15:0] spec_val;

    fp8_mul     u_mul (.a(mul_a), .b(mul_b), .fmt_a(fmt_a), .fmt_b(fmt_b), .p(m));
    bf16_add_s1 u_a1  (.a(c[ptag]), .b(prod), .sum(sum), .e_big(e_big),
                       .s_big(s_big), .spec(spec), .spec_val(spec_val));
    bf16_add_s2 u_a2  (.sum(q_sum), .e_big(q_e_big), .s_big(q_s_big),
                       .spec(q_spec), .spec_val(q_spec_val), .y(s));

    always @(posedge clk) begin
        if (!rst_n || clear) begin                       // R13, R14
            run <= 1'b0;  pv <= 1'b0;  qv <= 1'b0;  cnt <= 2'd0;
            c[0] <= 16'h0000; c[1] <= 16'h0000; c[2] <= 16'h0000; c[3] <= 16'h0000;
        end else begin
            pv <= run;
            qv <= pv;
            if (qv) c[qtag] <= s;                        // commit
            if (issue) begin
                run <= 1'b1;
                cnt <= 2'd0;
            end else if (run) begin
                cnt <= cnt + 2'd1;
                if (cnt == 2'd3) run <= 1'b0;
            end
        end
    end

    // data registers (not reset)
    always @(posedge clk) begin
        if (issue) begin
            oa0 <= a0; oa1 <= a1; ob0 <= b0; ob1 <= din;
        end
        if (run) begin
            prod <= m;
            ptag <= tag;
        end
        if (pv) begin
            q_sum <= sum; q_e_big <= e_big; q_s_big <= s_big;
            q_spec <= spec; q_spec_val <= spec_val;
            qtag <= ptag;
        end
    end

    assign busy = run | pv | qv;
    assign acc  = {c[3], c[2], c[1], c[0]};
endmodule

`default_nettype wire
