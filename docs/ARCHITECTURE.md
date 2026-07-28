# Echo Grid Ultrasonic OS — Architecture

## Core Principle

Computation = evolution of a coupled oscillator field  
Output = phase-mapped physical ultrasonic actuation

## Stack

1. **Field Kernel** (`echo_grid/core.py`)
   - 2D lattice of coupled oscillators
   - Discrete Laplacian coupling
   - External excitation (touch / sensors)

2. **Physics Mapper**
   - φ → frequency, amplitude, phase
   - Base frequency 40 kHz (typical ultrasonic transducers)

3. **Transport / Hardware Abstraction**
   - UDP packets (simulation / PC control)
   - SPI / parallel for FPGA
   - ESP-NOW for distributed ESP32 nodes

4. **Physical Layer**
   - MOSFET driver matrix
   - 40 kHz resonant transducers
   - Coherent interference field in air

## Physical Constraints

- Wavelength at 40 kHz ≈ 8.5 mm
- Emitter spacing ideally < λ/2 for good spatial control
- Phase coherence must be maintained at microsecond scale for strong interference patterns

## Extension Paths

- Closed-loop feedback (microphone array → field correction)
- Non-linear field dynamics (logic via interference)
- Multi-tile modular hardware (4×4 tiles → 16×16 arrays)

This repository is the complete software + firmware + FPGA reference for the system.
