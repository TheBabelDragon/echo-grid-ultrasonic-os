# Echo Grid Ultrasonic OS

Real-time programmable wavefield computer.  
Coupled oscillator field → ultrasonic mapping → physical body (optional).

Sibling to the optical body under the shared [Field Body Protocol](docs/FIELD_BODY_PROTOCOL.md).

## Run

```bash
pip install -r requirements.txt

# software only
python main.py

# live visualization (recommended)
python visualization/dashboard.py

# with physical Echo Body
python main.py --body
python visualization/dashboard.py --body --drive
```

## Live view

```bash
python visualization/dashboard.py
```

Shows:
- Phase field φ
- Frequency map (Hz)
- Live entropy / observation / time

## What you should see (CLI)

```
[field] mode=soft  entropy=0.412  obs=0.000  t=3.2s
✅ saved → echo_save.json
```

## Layout

```
echo_grid/          field kernel + body client
firmware/           ESP32 Echo Body
fpga/               Verilog DDS cores
visualization/      live field viewer
docs/               architecture + protocol
```

## License

MIT
