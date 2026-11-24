// bf16_add_s2: BF16 adder stage 2 of 2. Normalizes the aligned sum from
// bf16_add_s1 (carry right-shift or leading-zero left-shift), rounds RNE on the
// normalized 8-bit significand (as with an unbounded exponent), then applies
// overflow and FTZ on the rounded exponent and packs the result.
//
// Implements R3 (RNE every step), R8 (overflow to inf at |x| >= 2^128 - 2^119)
// and R9 (x + (-x) = +0; FTZ of results < 2^-126 with the sign kept).
`default_nettype none

module bf16_add_s2 (
    input  wire [11:0] sum,
    input  wire [7:0]  e_big,
    input  wire        s_big,
    input  wire        spec,
    input  wire [15:0] spec_val,
    output reg  [15:0] y
);
    // ---- normalize so the leading one lands at bit 10
    reg  [3:0]  lz;
    integer     i;
    always @* begin
        lz = 4'd11;
        for (i = 0; i <= 10; i = i + 1)
            if (sum[i]) lz = 4'd10 - i[3:0];
    end
    wire [10:0] norm = sum[11] ? {sum[11:2], sum[1] | sum[0]}
                               : (sum[10:0] << lz);
    wire signed [9:0] e_norm = $signed({2'b00, e_big})
                             + (sum[11] ? 10'sd1 : -$signed({6'd0, lz}));

    // ---- round to nearest even on 8-bit significand norm[10:3]
    wire        g   = norm[2];
    wire        st  = norm[1] | norm[0];
    wire        inc = g & (st | norm[3]);
    /* verilator lint_off UNUSEDSIGNAL */   // m_r[7] is the hidden bit
    wire [8:0]  m_r = {1'b0, norm[10:3]} + {8'd0, inc};
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [9:0] e_r = e_norm + (m_r[8] ? 10'sd1 : 10'sd0);
    wire [6:0]  frac = m_r[8] ? 7'd0 : m_r[6:0];

    always @* begin
        if (spec)
            y = spec_val;                          // R7, R8, R9 (from stage 1)
        else if (sum == 12'd0)
            y = 16'h0000;                          // R9: x + (-x) = +0
        else if (e_r >= 10'sd255)
            y = {s_big, 8'hFF, 7'd0};              // R8: overflow -> inf
        else if (e_r <= 10'sd0)
            y = {s_big, 15'd0};                    // R9: FTZ, sign kept
        else
            y = {s_big, e_r[7:0], frac};           // R3
    end
endmodule

`default_nettype wire
