// One cell of the output-stationary systolic array (Phase 5 design note §1-§2).
//
// In cycle t the cell sees A (with its tags) from the west and B from the
// north. If the A side is valid, the cell performs its MAC in cycle t: the
// accumulator takes acc + a*b (or a*b when the MAC carries `first`, the
// first k of a tile) at the clock edge that ends cycle t. A and the tags
// move east one cell per cycle, B moves south one cell per cycle.
module os_cell #(
    parameter int KW = 20
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 v_in,
    input  logic                 first_in,
    input  logic                 last_in,
    input  logic        [KW-1:0] k_in,
    input  logic signed [   7:0] a_in,
    input  logic signed [   7:0] b_in,
    output logic                 v_e,
    output logic                 first_e,
    output logic                 last_e,
    output logic        [KW-1:0] k_e,
    output logic signed [   7:0] a_e,
    output logic signed [   7:0] b_s,
    output logic signed [  31:0] acc
);
  logic signed [31:0] a32, b32, prod;
  always_comb begin
    a32  = {{24{a_in[7]}}, a_in};
    b32  = {{24{b_in[7]}}, b_in};
    prod = a32 * b32;  // |a*b| <= 2^14: exact in 32 bits
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      v_e     <= 1'b0;
      first_e <= 1'b0;
      last_e  <= 1'b0;
      k_e     <= '0;
      a_e     <= '0;
      b_s     <= '0;
      acc     <= '0;
    end else begin
      v_e     <= v_in;
      first_e <= first_in;
      last_e  <= last_in;
      k_e     <= k_in;
      a_e     <= a_in;
      b_s     <= b_in;
      if (v_in) acc <= first_in ? prod : acc + prod;
    end
  end
endmodule
