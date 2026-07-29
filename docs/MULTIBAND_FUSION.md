# Multiband / multi-node overlap elimination

## Idea

```
CSI sources (node × band)
        → per-source residual projection
        → agreement map (geometric mean)
        → shared RadioBeliefField (room + dynamic)
        → gated motion + tracks
        → φ inject
```

Dynamic mass is **trusted** when:

- ≥2 sources agree spatially, or
- ≥2 bands agree (2.4 / 5 / 6)

Single-source peaks are **damped** (clutter quarantine), not full track birth fuel.

## Packet fields

Optional on each CSI JSON packet:

```json
{
  "node": "esp32_node_01",
  "band": "2.4",
  "channel": 6,
  "csi": [...],
  "movement_intensity": 0.4
}
```

If `band` is omitted, channel heuristics apply; ESP default is **2.4**.

## Multi-node without 5 GHz

Two ESP32 nodes on 2.4 still exercise the **same consistency gate** (multi-source overlap elimination). True multiband lights up automatically when packets carry `band: "5"` / `"6"`.

## Logs

```
[multiband] sources=2 bands=1 agreed=True motion=0.41 conf=0.70
[CSI] ... agreed=True bands=1
```
