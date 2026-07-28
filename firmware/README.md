# Echo Body — Hardware-Ready Ultrasonic Body

Implements the [Field Body Protocol](../docs/FIELD_BODY_PROTOCOL.md) with a real sensing path.

## Sensing (closed-loop)

Default configuration uses **ESP32 ADC** on GPIO 34.

You can connect any of the following:

| Sensor type                    | Notes                                      |
|--------------------------------|--------------------------------------------|
| Electret microphone + preamp   | Simplest starting point                    |
| MEMS microphone (analog out)   | Cleaner, still analog                      |
| Ultrasonic receiver transducer | Same frequency family as the emitters      |
| Envelope detector              | Useful if you only care about amplitude    |

The `AcousticSensor` class averages multiple ADC samples, subtracts a baseline, and normalises to `[0, 1]`.  
Later you can replace its internals with I2S or an external ADC without touching the protocol or the host.

## Pin map (default)

```
Emitters (PWM):  GPIO 25, 26, 27, 14
Sensor (ADC):    GPIO 34
```

Change the constants at the top of `echo_body.ino` to match your board.

## Commands

```
EXCITE <id>     fire one emitter, then sample the sensor
MAP             recalibrate baseline + light one-hot response check
VERIFY          identity / calibration state
PASSIVE         return to background observation (emits OBS lines)
```

## Observation stream

In PASSIVE mode (and after EXCITE) the body emits:

```
OBS {"body_id":"echo_us_001","body_type":"ultrasonic", ... "observed":0.1834 ...}
```

The Python host already consumes these and folds them back into the field.

## Build

Arduino IDE or PlatformIO.  
Select an ESP32 / ESP32-S3 board, flash, open Serial at 115200.
