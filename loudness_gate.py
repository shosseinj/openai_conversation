"""Amplitude-only gate: never normalize or amplify audio."""
import numpy as np


def validate_threshold(value):
    value = float(value)
    if not np.isfinite(value) or not -120 <= value <= 0:
        raise ValueError('Minimum dBFS must be finite and between -120 and 0')
    return value


def measure(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Cannot measure empty or invalid audio')
    rms = float(np.sqrt(np.mean(audio * audio)))
    dbfs = float(20 * np.log10(rms)) if rms else float('-inf')
    return rms, dbfs


def passes(audio, minimum=-40):
    minimum = validate_threshold(minimum)
    rms, dbfs = measure(audio)
    accepted = rms > 0 and dbfs >= minimum
    print(f'Speech RMS: {rms:.6f} | dBFS: {dbfs:.2f} | minimum: {minimum:.2f} -> {"ACCEPT" if accepted else "IGNORE"}', flush=True)
    return accepted


def print_calibration(close, far):
    for label, values in [('Close voice', close), ('Distant speech', far)]:
        a = np.asarray(values)
        print(f'{label}: n={len(a)} | min={a.min():.2f} | median={np.median(a):.2f} | max={a.max():.2f} | p10={np.percentile(a,10):.2f} | p90={np.percentile(a,90):.2f} dBFS', flush=True)
    lower = np.percentile(far, 90)
    upper = np.percentile(close, 10)
    if upper > lower:
        print(f'Suggested starting threshold: {(upper+lower)/2:.2f} dBFS. Test and adjust.', flush=True)
    else:
        print('Levels overlap: no reliable threshold separates these recordings. Adjust microphone position/gain and recalibrate.', flush=True)
