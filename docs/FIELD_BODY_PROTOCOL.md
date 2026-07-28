# Field Body Protocol (v0.1 + observations)

Common contract between any physical body and a host.

## Commands (host → body)

```
EXCITE <id>
MAP
VERIFY
PASSIVE
```

## Observations (body → host)

Bodies emit lines beginning with `OBS ` followed by a JSON object:

```json
OBS {"body_id":"echo_us_001","body_type":"ultrasonic","excitation_id":-1,"geometry_state":"calibrated","health":"ok","regions":[{"region":"ambient","observed":0.12,"confidence":0.6}]}
```

Hosts should parse any line starting with `OBS ` and ignore the rest.

## Closed loop

1. Host may send `EXCITE`
2. Body acts and later emits `OBS …`
3. Host folds the observation back into its field / controller
4. Cycle continues

This is the minimal viable closed loop. Real sensors replace the observation stub without changing the protocol.
