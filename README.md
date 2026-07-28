# Echo Grid Ultrasonic OS

**A real-time programmable wavefield computer**  
Coupled oscillator field → ultrasonic phase/amplitude/frequency mapping → physical actuation.

This is a software-defined spatial acoustic actuator system designed for phased ultrasonic arrays and distributed emitter swarms.

It also serves as the **ultrasonic physical body** sibling to [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3) under a shared [Field Body Protocol](docs/FIELD_BODY_PROTOCOL.md).

## Quick Start (software only)

```bash
pip install -r requirements.txt
python main.py
```

## With physical Echo Body

```bash
# Flash firmware/echo_body.ino to an ESP32 first
python main.py --body                # auto-detect serial port
python main.py --body /dev/ttyUSB0   # explicit port
python main.py --body --drive        # also map the live field onto emitters
```

## Project Structure

```
echo-grid-ultrasonic-os/
├── echo_grid/
│   ├── core.py           # Field kernel + OS controller
│   ├── body_client.py    # Field Body Protocol client
│   └── __init__.py
├── firmware/
│   ├── echo_body.ino     # ESP32 ultrasonic body
│   └── README.md
├── fpga/                 # Verilog DDS cores
├── visualization/
├── docs/
│   ├── ARCHITECTURE.md
│   └── FIELD_BODY_PROTOCOL.md
├── main.py
└── requirements.txt
```

## Core Equation

$$
\phi_{t+1} = \phi_t + \lambda \nabla^2 \phi_t - \gamma \phi_t + \omega
$$

## License

MIT
