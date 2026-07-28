// Echo Grid Ultrasonic OS — Top Level FPGA Core
// 16×16 phase array with SPI control interface (stub)
// Expand phase_mem loading via SPI or AXI-lite in real designs

module echo_grid_top (
    input  wire         clk_100m,
    input  wire         rst_n,
    // Simple SPI slave ports (expand as needed)
    input  wire         spi_sclk,
    input  wire         spi_mosi,
    input  wire         spi_cs_n,
    output wire         spi_miso,
    // Flattened PWM outputs for 256 emitters
    output wire [255:0] pwm_out
);

    // Phase increment memory (one word per emitter)
    reg [31:0] phase_mem [0:15][0:15];

    // Instantiate 16×16 DDS channels
    genvar gx, gy;
    generate
        for (gy = 0; gy < 16; gy = gy + 1) begin : ROW
            for (gx = 0; gx < 16; gx = gx + 1) begin : COL
                wire pwm;

                echo_dds_channel u_channel (
                    .clk       (clk_100m),
                    .rst_n     (rst_n),
                    .phase_inc (phase_mem[gy][gx]),
                    .pwm_out   (pwm)
                );

                assign pwm_out[gy*16 + gx] = pwm;
            end
        end
    endgenerate

    // TODO: Add SPI packet decoder to write phase_mem
    // Packet format example: {cmd[15:0], x[7:0], y[7:0], phase[31:0]}

    // Dummy MISO for now
    assign spi_miso = 1'b0;

endmodule
