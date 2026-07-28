# Capture & replay

## Record (with labels)

```bash
python tools/capture_csi.py -o data/session.jsonl
```

Keyboard: `e` empty · `1`/`2`/`3` persons · `w` walk · `s` still · `q` quit

## Replay into live dashboard

Terminal A:
```bash
python visualization/dashboard.py
```

Terminal B:
```bash
python tools/replay_csi.py data/session.jsonl --loop --rate 15
```

## Multi-node

Bridge keeps a registry of recent `node` ids (5s TTL). Status / closed-loop reports `nodes=N`.
