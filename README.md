# Echo Grid Ultrasonic OS

Real-time programmable wavefield computer.  
Coupled oscillator field → ultrasonic mapping → physical body (optional).

Sibling to the optical body under the shared [Field Body Protocol](docs/FIELD_BODY_PROTOCOL.md).

## Run

```bash
pip install -r requirements.txt

# software only (stable, production defaults)
python main.py

# with physical Echo Body
python main.py --body
python main.py --body /dev/ttyUSB0 --drive
```

## What you should see

```
[field] mode=soft  entropy=0.412  obs=0.000  t=3.2s
✅ saved → echo_save.json
```

- `mode=soft` — no body attached  
- `mode=body` — closed-loop body connected  
- `entropy` stays bounded (soft clamp)  
- `obs` rises only when a real sensor reports energy  
- saves happen once per interval, not in a spam burst

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
