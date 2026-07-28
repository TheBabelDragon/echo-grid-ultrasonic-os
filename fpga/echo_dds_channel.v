// Echo Grid Ultrasonic OS — Single Channel DDS
// Phase accumulator + PWM output for one ultrasonic emitter

module echo_dds_channel (
    input  wire        clk,        // system clock (recommend 100–200 MHz)
    input  wire        rst_n,      // active-low reset
    input  wire [31:0] phase_inc,  // phase increment (from field mapper)
    output wire        pwm_out     // square-wave ultrasonic drive
);

    reg [31:0] phase_acc = 32'd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            phase_acc <= 32'd0;
        else
            phase_acc <= phase_acc + phase_inc;
    end

    // MSB of accumulator = 50% duty cycle PWM
    assign pwm_out = phase_acc[31];

endmodule
