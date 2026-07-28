# Actuator equilibrium + closed loop

## What panel 3 is

`|Δf|` is the **virtual ultrasonic drive map**:

```
f(x,y) = 40 kHz + k · φ(x,y)
```

It is what emitters *would* run. It is **not** a live measurement from transducers unless a body is attached.

## Loops you have today

```
CSI ESP  --4210-->  tracks / motion
                 -->  inject into φ
                 -->  |Δf| map (virtual actuators)
Echo Grid --4211-->  boost / quiet / field telemetry → CYD
```

That is **RF closed-loop** (sense RF → field → command sensing rate).

## Equilibrium under CSI only

Steady motion → steady inject → inject balances damping → non-zero `|Δf|` plateaus (`drive=` stays up).

Quiet room → inject stops → field settles → `|Δf|` goes dark.

**You are not missing a flag** for that. You need **live CSI packets** (ESP on same LAN).

## Full *acoustic* actuator closed loop (optional extra input)

| Input | Role |
|-------|------|
| CSI (default) | RF motion → field |
| `--body` | ultrasonic mic/body observations → field |
| `--body --drive` | field → excite physical emitters |

Without transducers + body firmware, panel 3 stays a **planned** drive map, not a measured acoustic equilibrium.

## Run (CSI standard)

```bash
python visualization/dashboard.py
# same as old: python visualization/dashboard.py --csi
```
