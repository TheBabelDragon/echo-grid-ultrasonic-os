# FPGA Cores for Echo Grid

## Files
- `echo_dds_channel.v` — single emitter phase accumulator + PWM
- `echo_grid_top.v` — 16×16 array top module

## Recommended Clock
100–200 MHz for good frequency resolution at 40 kHz base.

## Phase Increment Calculation (software side)
```python
phase_inc = int((freq / clk_freq) * (2**32))
```

## Next Steps for Real Hardware
1. Add SPI / AXI-lite register map for live phase updates
2. Add amplitude control (PWM duty or external DAC)
3. Clock domain crossing and PLL for low-jitter distribution
4. Timing constraints for phase coherence across the array
