// bf16_add_s1: BF16 adder stage 1 of 2. Resolves special values, orders the
// operands by magnitude, aligns the smaller significand with guard/round/sticky
// bits and adds or subtracts. Stage 2 (bf16_add_s2) normalizes, rounds and
// packs. bf16_add chains the two combinationally; the datapath registers the
// 38-bit bundle between them to meet 20 ns at the ss corner.
//
// Implements R7 (NaN in / +inf + -inf -> canonical NaN), R8 (inf rules) and
// the R9 zero cases that don't depend on the sum (-0 only from -0 + -0).
// Subnormal inputs are read as zero.
`default_nettype none

module bf16_add_s1 (
    input  wire [15:0] a,          // accumulator
    input  wire [15:0] b,          // product
    output wire [11:0] sum,        // aligned add/sub result, bit 11 = carry
    output wire [7:0]  e_big,      // exponent of the larger operand
    output wire        s_big,      // sign of the larger operand
    output reg         spec,       // result fully determined by special cases
    output reg  [15:0] spec_val    // ...and that result
);
    // ---- unpack
    wire       sa = a[15], sb = b[15];
    wire [7:0] ea = a[14:7], eb = b[14:7];
    wire       a_nan  = (ea == 8'hFF) && (a[6:0] != 7'd0);
    wire       b_nan  = (eb == 8'hFF) && (b[6:0] != 7'd0);
    wire       a_inf  = (ea == 8'hFF) && (a[6:0] == 7'd0);
    wire       b_inf  = (eb == 8'hFF) && (b[6:0] == 7'd0);
    wire       a_zero = (ea == 8'h00);      // zero or subnormal: reads as zero
    wire       b_zero = (eb == 8'h00);

    // ---- order by magnitude (ties: either order gives the same result)
    wire       swap = {eb, b[6:0]} > {ea, a[6:0]};
    assign     s_big = swap ? sb : sa;
    assign     e_big = swap ? eb : ea;
    wire [7:0] e_sml = swap ? ea : eb;
    wire [7:0] m_big = {1'b1, swap ? b[6:0] : a[6:0]};
    wire       sml_zero = swap ? a_zero : b_zero;
    wire [7:0] m_sml_v  = sml_zero ? 8'd0 : {1'b1, swap ? a[6:0] : b[6:0]};
    wire       eff_sub = sa ^ sb;

    // ---- align: 8-bit significand + guard, round, sticky = 11 bits
    wire [7:0]  d      = e_big - e_sml;
    wire [18:0] sml_x  = {m_sml_v, 11'd0} >> ((d > 8'd11) ? 5'd11 : d[4:0]);
    wire [10:0] sml_al = {sml_x[18:9], |sml_x[8:0]};   // last bit is sticky
    wire [10:0] big_al = {m_big, 3'b000};

    // ---- add / subtract (bit 11 = carry)
    assign sum = eff_sub ? ({1'b0, big_al} - {1'b0, sml_al})
                         : ({1'b0, big_al} + {1'b0, sml_al});

    // ---- special-value results
    always @* begin
        spec = 1'b1;
        if (a_nan || b_nan || (a_inf && b_inf && (sa != sb)))
            spec_val = 16'h7FC0;                   // R7
        else if (a_inf)
            spec_val = a;                          // R8: inf + finite / same inf
        else if (b_inf)
            spec_val = b;                          // R8
        else if (a_zero && b_zero)
            spec_val = {sa & sb, 15'd0};           // R9: -0 only from -0 + -0
        else begin
            spec = 1'b0;
            spec_val = 16'h0000;
        end
    end
endmodule

`default_nettype wire
