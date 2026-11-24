// fp8_unpack: decode an E4M3FN/E5M2 byte into sign, class flags and a
// normalized 4-bit significand with unbiased exponent (R4, R5).
// value = sig/8 * 2^exp for finite nonzero inputs.
`default_nettype none

module fp8_unpack (
    input  wire [7:0] x,
    input  wire       fmt,      // 0 = E4M3FN, 1 = E5M2
    output wire       sign,
    output wire       is_zero,
    output wire       is_inf,
    output wire       is_nan,
    output reg  [3:0] sig,      // normalized 1.xxx (bit 3 set) unless zero
    output reg  signed [6:0] exp // unbiased exponent of sig/8
);
    // Raw fields, aligned to a common 4-bit significand {lead, f2, f1, f0}.
    wire [4:0] e_raw   = fmt ? x[6:2] : {1'b0, x[6:3]};
    wire [2:0] frac    = fmt ? {x[1:0], 1'b0} : x[2:0];
    wire       e_zero  = (e_raw == 5'd0);
    wire       e_max   = fmt ? (x[6:2] == 5'h1F) : 1'b0;

    assign sign    = x[7];
    // R4: E4M3FN NaN is S.1111.111 only; no infinities.
    // R5: E5M2 exponent all-ones: mantissa 0 -> inf, else NaN.
    assign is_nan  = fmt ? (e_max && x[1:0] != 2'b00) : (x[6:0] == 7'h7F);
    assign is_inf  = e_max && (x[1:0] == 2'b00);
    assign is_zero = e_zero && (frac == 3'b000);

    // bias 7 (E4M3FN) or 15 (E5M2); subnormals use exponent 1 - bias.
    wire signed [6:0] bias    = fmt ? 7'sd15 : 7'sd7;
    wire signed [6:0] e_unb   = $signed({2'b00, e_raw}) - bias;
    wire signed [6:0] e_sub   = 7'sd1 - bias;

    always @* begin
        if (!e_zero) begin
            sig = {1'b1, frac};
            exp = e_unb;
        end else if (frac[2]) begin          // subnormal: normalize (R4, R5)
            sig = {frac, 1'b0};
            exp = e_sub - 7'sd1;
        end else if (frac[1]) begin
            sig = {frac[1:0], 2'b00};
            exp = e_sub - 7'sd2;
        end else begin
            sig = {frac[0], 3'b000};
            exp = e_sub - 7'sd3;
        end
    end
endmodule

`default_nettype wire
