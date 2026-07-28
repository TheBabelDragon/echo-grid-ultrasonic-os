# CSI ↔ Echo Grid closed loop

```
ESP CSI  --UDP:4210-->  Echo Grid field / tracks
ESP CSI  <--UDP:4211--  Echo Grid commands
```

## Commands (`type: echo_cmd`)

| cmd | fields | effect on ESP |
|-----|--------|----------------|
| `boost` | `level` 0..1 | faster rate + higher motion gain |
| `quiet` | | slow rate, boost=0 |
| `set_rate` | `interval_ms` | fixed send interval |
| `ping` | | serial log only |

## Host policy

- many tracks / high motion / high entropy → **boost**
- idle field → **quiet**
- otherwise → adaptive **set_rate**

## Flash ESP

```bash
cd wifi-sensing-system/esp32
git pull
pio run -e esp32-standard -t upload --upload-port /dev/ttyUSB0
```

Serial should show `[CMD] boost` / `quiet` when Echo Grid is running with `--csi`.
