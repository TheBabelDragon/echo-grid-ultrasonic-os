# CSI Integration

Echo Grid can take live WiFi CSI as an external field excitation.

## Source

Compatible with [wifi-sensing-system](https://github.com/TheBabelDragon/wifi-sensing-system):

- ESP32 firmware: `esp32/esp32_csi_udp_sender.ino`
- UDP port **4210**
- JSON packet with `csi` array + `rssi`

## Run

```bash
# software field + CSI listener
python main.py --csi
python visualization/dashboard.py --csi

# CSI only (no synthetic demo drive)
python visualization/dashboard.py --csi --no-demo

# full stack
python visualization/dashboard.py --csi --body --drive
```

## Mapping

1. CSI amplitude vector is compared to a slow baseline
2. Mean absolute deviation → motion energy `csi ∈ [0, 1]`
3. Subcarrier left/right balance biases injection `(x, y)`
4. Energy is injected into the Echo field each step

When people move through the RF field, the ultrasonic wavefield responds.
