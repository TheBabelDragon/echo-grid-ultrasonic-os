# Echo Grid Ultrasonic OS

Real-time programmable wavefield computer.  
Coupled oscillator field → ultrasonic mapping → physical body + optional **WiFi CSI** input.

## Run

```bash
pip install -r requirements.txt

# software only
python main.py

# live visualization
python visualization/dashboard.py

# interactive with WiFi CSI (from wifi-sensing-system nodes)
python visualization/dashboard.py --csi

# CSI + ultrasonic body
python visualization/dashboard.py --csi --body --drive
```

CSI listens on **UDP 4210** — same contract as the wifi-sensing-system ESP32 nodes.

See [docs/CSI_INTEGRATION.md](docs/CSI_INTEGRATION.md).

## License

MIT
