// mantis_stage: command decode, operand staging and format registers.
//
// Shared front end of both T7 datapath organizations. Implements the
// staging part of R13 (CLEAR loads fmt, discards partial groups), R14 (reset),
// R15/P5/P6 (only accepted WRITE/CLEAR change n, slots, fmt).
//
// A WRITE to slot 3 asserts `issue` combinationally in the cycle it is
// presented; the step's operands are slot0..2 (registered) plus `din` (B[k][1]).
// Consumers sample issue/clear at the same rising edge.
`default_nettype none

module mantis_stage (
    input  wire       clk,
    input  wire       rst_n,     // synchronous, active low
    input  wire       in_valid,
    input  wire [1:0] cmd,
    input  wire [7:0] din,
    output reg  [7:0] a0,        // slot 0: A[0][k]
    output reg  [7:0] a1,        // slot 1: A[1][k]
    output reg  [7:0] b0,        // slot 2: B[k][0]
    output wire       issue,     // this edge completes a group (B[k][1] = din)
    output wire       clear,     // this edge is an accepted CLEAR
    output reg        fmt_a,
    output reg        fmt_b,
    output reg  [1:0] n
);
    localparam [1:0] WRITE = 2'b00, CLEAR = 2'b01;

    wire write = in_valid && (cmd == WRITE);
    assign clear = in_valid && (cmd == CLEAR);
    assign issue = write && (n == 2'd3);

    always @(posedge clk) begin
        if (!rst_n) begin                 // R14
            n     <= 2'd0;
            fmt_a <= 1'b0;
            fmt_b <= 1'b0;
        end else if (clear) begin         // R13
            n     <= 2'd0;
            fmt_a <= din[0];
            fmt_b <= din[1];
        end else if (write) begin
            n <= n + 2'd1;                // slot 3 wraps to 0 (issue)
        end
    end

    // Operand slots: data registers, not reset (a partial group never issues).
    always @(posedge clk) begin
        if (rst_n && write && !clear) begin
            case (n)
                2'd0:    a0 <= din;
                2'd1:    a1 <= din;
                2'd2:    b0 <= din;
                default: ;                // slot 3 is consumed directly
            endcase
        end
    end
endmodule

`default_nettype wire
