// fp8_mul: exact FP8 x FP8 product as a BF16 bit pattern (gate T5; used by
// mantis_dp_shared).
//
// Implements R2 (exact product), R4 (E4M3FN decode), R5 (E5M2 decode),
// R6 (independent formats), R7 (NaN, inf*0), R8 (signed inf), R9 (signed zero).
// Purely combinational; any pipelining is decided at the datapath level (T7).
//
// Every finite product has <= 8 significant bits (4b x 4b significands) and an
// unbiased exponent in [-32, 32], so the BF16 result is exact (R2).
`default_nettype none

module fp8_mul (
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    input  wire        fmt_a,   // R6
    input  wire        fmt_b,   // R6
    output reg  [15:0] p        // BF16
);
    wire       sa, za, ia, na, sb, zb, ib, nb;
    wire [3:0] siga, sigb;
    wire signed [6:0] ea, eb;

    fp8_unpack ua (.x(a), .fmt(fmt_a), .sign(sa), .is_zero(za), .is_inf(ia),
                   .is_nan(na), .sig(siga), .exp(ea));
    fp8_unpack ub (.x(b), .fmt(fmt_b), .sign(sb), .is_zero(zb), .is_inf(ib),
                   .is_nan(nb), .sig(sigb), .exp(eb));

    wire       s    = sa ^ sb;
    wire [7:0] prod = siga * sigb;        // in [64, 225]: lead at bit 7 or 6
    wire       top  = prod[7];
    // value = prod/64 * 2^(ea+eb); renormalize to 1.xxxxxxx
    wire signed [7:0] e_res = ea + eb + $signed({7'b0, top});  // keep all-signed
    wire [6:0] mant = top ? prod[6:0] : {prod[5:0], 1'b0};
    wire [7:0] e_bf = e_res + 8'sd127;    // 95..160 for finite inputs

    always @* begin
        if (na || nb || (ia && zb) || (za && ib))
            p = 16'h7FC0;                 // R7: canonical NaN
        else if (ia || ib)
            p = {s, 8'hFF, 7'h00};        // R8: signed infinity
        else if (za || zb)
            p = {s, 15'h0000};            // R9: zero with XOR sign
        else
            p = {s, e_bf, mant};          // R2: exact
    end
endmodule

`default_nettype wire
