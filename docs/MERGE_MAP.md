# Merge map: Echo Grid · optical body · MetaField

## Three different things

| Repo | What it is |
|------|------------|
| **echo-grid-ultrasonic-os** | Real-time field OS + CSI + virtual 40 kHz map |
| **optical-body-s3** | Physical **light** body (laser + BPW34), Field Body Protocol |
| **metafield** | Lattice QCD / gauge-field research code (PyTorch) — **not** the optical body |

## What we merged

**Echo Grid host ↔ optical-body-s3** via shared **Field Body Protocol**:

```
CSI nodes          → φ (RF motion)
optical-body-s3    → φ (light observations)   OBS / JSON serial
Echo Grid          → optical-body EXCITE/MAP   when --drive
Echo Grid          → CSI nodes                 :4211 commands
```

Same φ lattice. Optical and RF are **two sensors into one field**, not two competing OSes.

## What we did *not* merge

**metafield** (QCD HMC / Dirac) stays its own research stack. Do not run it as the Echo Grid body.

## How to run the merge

1. Flash / power **optical-body-s3** USB serial  
2. CSI ESP nodes on LAN  
3. Host:

```bash
cd echo-grid-ultrasonic-os
pip install pyserial   # if needed
python visualization/dashboard.py --body
# closed optical loop:
python visualization/dashboard.py --body --drive
```

Status line should show `body=optical` (or body type from device) and `obs=` moving with light.
