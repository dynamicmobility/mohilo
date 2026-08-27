"""Peripherals of the host machine running an experiment."""

import numpy as np


def chime(volume: float = 1.0, sample_rate: int = 44100):
    """Two-tone chime out the host's default output device. Volume from 0 to 1."""
    import sounddevice as sd

    def tone(freq, dur, decay=7.0):
        t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
        env = np.exp(-decay * t)
        env[:200] *= np.linspace(0, 1, 200)  # attack ramp, prevents click
        wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        return wave * env

    signal = np.concatenate([tone(880, 0.18), tone(1318.5, 0.6)]) * volume
    sd.play(signal, samplerate=sample_rate)
    sd.wait()
