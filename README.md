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

# emit FieldObservation JSONL for MetaField
python visualization/dashboard.py --csi --metafield-log /tmp/metafield/echo.jsonl
```

CSI listens on **UDP 4210** — same contract as the wifi-sensing-system ESP32 nodes.

See [docs/CSI_INTEGRATION.md](docs/CSI_INTEGRATION.md).

## MetaField bridge

Echo can publish canonical `FieldObservation` packets (same schema as optical-body-s3):

```bash
# Echo side
python visualization/dashboard.py --csi --metafield-log /tmp/metafield/echo.jsonl

# MetaField side (same consumer as optical)
cd ../metafield
source .venv/bin/activate
python optical_serial_consumer.py \
  --file /tmp/metafield/echo.jsonl \
  --save /tmp/metafield/field_memory.jsonl
```

Regions emitted: `motion`, `entropy`, `df_max`, `drive`, `fuse`, plus `track_<id>` for each active track.

## CAVITATION.SWEEP (calibration)

Measurement-first cavitation calibration. Characterises abstract drive command
vs acoustic/optical response. Simulation is the default; no hardware is
energised unless an explicit sensor backend is supplied.

```bash
# hardware-free demo
python tools/cavitation_sweep_demo.py

# tests
python -m pytest tests/test_cavitation.py -v
```

See [docs/CAVITATION_SWEEP.md](docs/CAVITATION_SWEEP.md).

## License

MIT
