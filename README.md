# Echo Grid Ultrasonic OS

**A real-time programmable wavefield computer**  
Coupled oscillator field → ultrasonic phase/amplitude/frequency mapping → physical actuation.

This is a software-defined spatial acoustic actuator system designed for phased ultrasonic arrays and distributed emitter swarms.

## What it is

- **Field Kernel**: Continuous 2D coupled oscillator lattice (the computational brain)
- **Mapper**: Deterministic translation of field state into ultrasonic parameters
- **Hardware Abstraction**: UDP / SPI / FPGA / ESP32 ready
- **Physical Output**: Interference fields, pressure nodes, localized force (when hardware is attached)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Optional visualization:
```bash
python visualization/dashboard.py
```

## Project Structure

```
echo-grid-ultrasonic-os/
├── echo_grid/           # Core Python OS
│   ├── __init__.py
│   └── core.py           # Field + Mapper + Controller
├── fpga/               # Verilog cores
│   ├── echo_dds_channel.v
│   └── echo_grid_top.v
├── firmware/           # ESP32 swarm
│   └── esp32_swarm.ino
├── visualization/      # Live field viewer
│   └── dashboard.py
├── docs/
│   └── ARCHITECTURE.md
├── main.py
├── requirements.txt
└── LICENSE
```

## Core Equation

$$
\phi_{t+1} = \phi_t + \lambda \nabla^2 \phi_t - \gamma \phi_t + \omega
$$

Where:
- $\phi$ = field state
- $\lambda$ = coupling strength
- $\gamma$ = damping
- $\omega$ = external excitation

## Ultrasonic Mapping

```
freq  = 40000 + k * φ
amp   = |φ|
phase = φ
```

## Hardware Paths

1. **Simulation only** → run `main.py`
2. **ESP32 swarm** → flash `firmware/esp32_swarm.ino`
3. **FPGA coherent array** → synthesize `fpga/echo_grid_top.v`

## License

MIT License — see [LICENSE](LICENSE)

---

Built as a complete, self-contained reference implementation of a field-native ultrasonic computing system.
