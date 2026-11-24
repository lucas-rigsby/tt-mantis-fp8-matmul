// tt_um_lucas_rigsby_mantis: MANTIS 2x2 FP8 -> BF16 matrix-multiply accelerator,
// Tiny Tapeout top level (docs/spec.md rev 0.6, baseline without JTAG).
//
// Pins (spec Interfaces):
//   ui_in[7:0]   operand byte (WRITE) / format bits (CLEAR)
//   uio_in[0]    in_valid      uio_in[2:1]  cmd
//   uio_in[3,5,6] reserved JTAG tck/tms/tdi: inert (R25)
//   uo_out[7:0]  result byte selected by read pointer p (R12)
//   uio_out[4]   busy (R11)    uio_out[7]  reserved JTAG tdo = 0 (R25)
//   uio_oe       constant 8'h90 (R15, P1)
//
// Implements R12 (READ advances p mod 8, byte map, uo_out combinational from
// registered p), R13/R14 (p cleared by CLEAR and reset), R15 (static pins,
// idle cycles), R25 (JTAG pins inert), P1, P5, P6. Arithmetic, staging, busy
// and CLEAR flushing live in mantis_dp_shared.
`default_nettype none

module tt_um_lucas_rigsby_mantis (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,      // always 1 when selected; unused
    input  wire       clk,
    input  wire       rst_n
);
    localparam [1:0] CLEAR = 2'b01, READ = 2'b10;

    wire        in_valid = uio_in[0];
    wire [1:0]  cmd      = uio_in[2:1];
    wire        busy;
    wire [63:0] acc;            // {C11, C10, C01, C00}
    wire        fmt_a, fmt_b;
    wire [1:0]  n;

    mantis_dp_shared u_dp (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .cmd(cmd), .din(ui_in),
        .busy(busy), .acc(acc), .fmt_a(fmt_a), .fmt_b(fmt_b), .n(n)
    );

    // ---- read pointer (R12, R13, R14, P5)
    reg [2:0] p;
    always @(posedge clk) begin
        if (!rst_n)
            p <= 3'd0;
        else if (in_valid && cmd == CLEAR)
            p <= 3'd0;
        else if (in_valid && cmd == READ)
            p <= p + 3'd1;                     // wraps 7 -> 0
    end

    // ---- result byte: p = 2*word + half, little-endian within a word (R12)
    wire [15:0] word = acc[{p[2:1], 4'b0000} +: 16];
    assign uo_out  = p[0] ? word[15:8] : word[7:0];

    // ---- static pins (R15, R25, P1)
    assign uio_out = {1'b0, 2'b00, busy, 4'b0000};   // [7]=tdo 0, [4]=busy
    assign uio_oe  = 8'h90;

    // JTAG pins uio_in[3,5,6] and output-only pins uio_in[4,7] are inert (R25).
    wire _unused = &{ena, uio_in[7:3], fmt_a, fmt_b, n, 1'b0};

`ifdef FORMAL
    // P1: static pins; P5: p changes only on accepted READ/CLEAR or reset;
    // P6: NOP / invalid cycles leave p unchanged.
    reg f_past_valid = 1'b0;
    always @(posedge clk) f_past_valid <= 1'b1;
    always @(*) begin
        assert (uio_oe == 8'h90);                        // P1
        assert (uio_out[7:5] == 3'b000);                 // P1, R25
        assert (uio_out[3:0] == 4'b0000);                // P1
        assert (uio_out[4] == busy);
    end
    always @(posedge clk) if (f_past_valid) begin
        if (!$past(rst_n)) begin
            assert (p == 3'd0);                          // R14
        end else if ($past(in_valid) && $past(cmd) == CLEAR) begin
            assert (p == 3'd0);                          // R13
        end else if ($past(in_valid) && $past(cmd) == READ) begin
            assert (p == $past(p) + 3'd1);               // R12
        end else begin
            assert (p == $past(p));                      // P5, P6
        end
        if ($past(rst_n) && !($past(in_valid) && ($past(cmd) == 2'b00 || $past(cmd) == CLEAR)))
            assert (n == $past(n));                      // P5, P6: n only on WRITE/CLEAR
        // non-vacuity: the asserted transitions are reachable
        cover ($past(rst_n) && $past(p) == 3'd7 && p == 3'd0
               && $past(in_valid) && $past(cmd) == READ);    // wrap 7 -> 0
        cover ($past(rst_n) && $past(p) != 3'd0 && p == 3'd0
               && $past(in_valid) && $past(cmd) == CLEAR);   // CLEAR resets p
        cover (n == 2'd3 && busy);                           // partial group while busy
    end
`endif
endmodule

`default_nettype wire
