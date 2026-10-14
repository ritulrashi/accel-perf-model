// Output-stationary INT8 systolic array, R x C (Phase 5 design note §1-§2):
// the RTL of the Phase 2 §2.2 cycle contract.
//
//   feed:   in cycle f+k the edge sees a_col[0..R) (A column k) and b_row[0..C)
//           (B row k) with in_valid; tags in_first (k = 0 of the tile's first
//           chunk), in_last (last k of its last chunk), in_k (trace only)
//   skew:   row i's A (and tags) is delayed i cycles, column j's B j cycles,
//           so MAC(i, j, k) happens in cycle f + k + i + j
//   drain:  after the MAC carrying `last` at cell (R-1, C-1) (cycle
//           f_L + Kt + R + C - 3), R drain cycles show one accumulator row
//           each, row R-1 first; the tile is done the cycle after
//   error:  sticky; a `first` feed while a tile is still open (not yet fully
//           drained), or a non-first feed with no tile open
//   trace:  mac_valid / mac_k: which cell performs a MAC this cycle, with
//           which k (read-only, nothing feeds back)
module os_array #(
    parameter int R  = 4,
    parameter int C  = 16,
    parameter int KW = 20
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         in_valid,
    input  logic                         in_first,
    input  logic                         in_last,
    input  logic        [        KW-1:0] in_k,
    input  logic signed [           7:0] a_col     [R],
    input  logic signed [           7:0] b_row     [C],
    output logic                         drain_valid,
    output logic [((R > 1) ? $clog2(R) : 1)-1:0] drain_row,
    output logic signed [          31:0] drain_data[C],
    output logic                         error,
    output logic                         mac_valid [R][C],
    output logic        [        KW-1:0] mac_k     [R][C]
);
  localparam int RW = (R > 1) ? $clog2(R) : 1;  // drain row index width
  localparam int DW = $clog2(R + 1);            // drain countdown width (0..R)
  localparam int AW = 3 + KW + 8;               // {valid, first, last, k, a}

  // ---------------------------------------------------------------- west edge: A + tags, row i delayed i cycles
  logic [AW-1:0] west[R];
  for (genvar i = 0; i < R; i++) begin : g_west
    if (i == 0) begin : g_d0
      assign west[i] = {in_valid, in_first, in_last, in_k, a_col[i]};
    end else begin : g_dn
      logic [AW-1:0] sr[i];
      always_ff @(posedge clk) begin
        if (rst) begin
          for (int d = 0; d < i; d++) sr[d] <= '0;
        end else begin
          sr[0] <= {in_valid, in_first, in_last, in_k, a_col[i]};
          for (int d = 1; d < i; d++) sr[d] <= sr[d-1];
        end
      end
      assign west[i] = sr[i-1];
    end
  end

  // ---------------------------------------------------------------- north edge: B, column j delayed j cycles
  logic signed [7:0] north[C];
  for (genvar j = 0; j < C; j++) begin : g_north
    if (j == 0) begin : g_d0
      assign north[j] = b_row[j];
    end else begin : g_dn
      logic signed [7:0] sr[j];
      always_ff @(posedge clk) begin
        if (rst) begin
          for (int d = 0; d < j; d++) sr[d] <= '0;
        end else begin
          sr[0] <= b_row[j];
          for (int d = 1; d < j; d++) sr[d] <= sr[d-1];
        end
      end
      assign north[j] = sr[j-1];
    end
  end

  // ---------------------------------------------------------------- cell grid
  // Cell (i, j) inputs; the east/south outputs of the last column/row are the
  // inputs of a column/row that does not exist (kept so every cell is identical).
  logic v_i[R][C+1], f_i[R][C+1], l_i[R][C+1];
  logic [KW-1:0] k_i[R][C+1];
  logic signed [7:0] a_i[R][C+1];
  logic signed [7:0] b_i[R+1][C];
  logic signed [31:0] acc[R][C];

  for (genvar i = 0; i < R; i++) begin : g_row
    assign {v_i[i][0], f_i[i][0], l_i[i][0], k_i[i][0], a_i[i][0]} = west[i];
  end
  for (genvar j = 0; j < C; j++) begin : g_col
    assign b_i[0][j] = north[j];
  end
  for (genvar i = 0; i < R; i++) begin : g_r
    for (genvar j = 0; j < C; j++) begin : g_c
      os_cell #(.KW(KW)) u_cell (
          .clk     (clk),
          .rst     (rst),
          .v_in    (v_i[i][j]),
          .first_in(f_i[i][j]),
          .last_in (l_i[i][j]),
          .k_in    (k_i[i][j]),
          .a_in    (a_i[i][j]),
          .b_in    (b_i[i][j]),
          .v_e     (v_i[i][j+1]),
          .first_e (f_i[i][j+1]),
          .last_e  (l_i[i][j+1]),
          .k_e     (k_i[i][j+1]),
          .a_e     (a_i[i][j+1]),
          .b_s     (b_i[i+1][j]),
          .acc     (acc[i][j])
      );
      assign mac_valid[i][j] = v_i[i][j];
      assign mac_k[i][j]     = k_i[i][j];
    end
  end

  // ---------------------------------------------------------------- drain and protocol check
  logic          corner_last;  // the tile's final MAC happens this cycle
  logic [DW-1:0] drain_left;   // drain rows still to show, counting this cycle's
  logic          tile_open;    // a tile has been fed and is not yet fully drained
  assign corner_last = v_i[R-1][C-1] && l_i[R-1][C-1];
  assign drain_valid = drain_left != '0;
  assign drain_row   = RW'(drain_left - DW'(1));  // row R-1 first
  always_comb begin
    for (int j = 0; j < C; j++) drain_data[j] = drain_valid ? acc[drain_row][j] : '0;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      drain_left <= '0;
      tile_open  <= 1'b0;
      error      <= 1'b0;
    end else begin
      if (corner_last) drain_left <= DW'(R);
      else if (drain_left != '0) drain_left <= drain_left - DW'(1);
      if (in_valid && in_first) tile_open <= 1'b1;
      else if (drain_left == DW'(1)) tile_open <= 1'b0;
      if (in_valid && (in_first == tile_open)) error <= 1'b1;
      if (corner_last && drain_left != '0) error <= 1'b1;
    end
  end

  // The outputs past the last column/row are unused by construction; fold
  // them into nothing observable without a lint waiver.
  logic unused_edge;
  always_comb begin
    unused_edge = 1'b0;
    for (int i = 0; i < R; i++)
      unused_edge = unused_edge ^ v_i[i][C] ^ f_i[i][C] ^ l_i[i][C] ^ (^k_i[i][C]) ^ (^a_i[i][C]);
    for (int j = 0; j < C; j++) unused_edge = unused_edge ^ (^b_i[R][j]);
  end
endmodule
