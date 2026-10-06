"""Stateful DeepFilterNet LADSPA runtime (official v0.5.6 Linux release).

Processes mono 48 kHz in 10 ms frames. No per-chunk model loading.
"""
import ctypes as C
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter


class Descriptor(C.Structure):
    _fields_ = [
        ('unique_id', C.c_ulong), ('label', C.c_char_p), ('properties', C.c_int),
        ('name', C.c_char_p), ('maker', C.c_char_p), ('copyright', C.c_char_p),
        ('port_count', C.c_ulong), ('port_descriptors', C.POINTER(C.c_int)),
        ('port_names', C.POINTER(C.c_char_p)), ('port_hints', C.c_void_p),
        ('implementation', C.c_void_p), ('instantiate', C.c_void_p),
        ('connect_port', C.c_void_p), ('activate', C.c_void_p), ('run', C.c_void_p),
        ('run_adding', C.c_void_p), ('set_gain', C.c_void_p),
        ('deactivate', C.c_void_p), ('cleanup', C.c_void_p),
    ]


class NoiseSuppressor:
    sample_rate = 48000
    frame_size = 480

    def __init__(self):
        path = Path(__file__).resolve().parent / 'deepfilter' / 'libdeep_filter_ladspa.so'
        self.library = C.CDLL(str(path))
        self.library.ladspa_descriptor.argtypes = [C.c_ulong]
        self.library.ladspa_descriptor.restype = C.POINTER(Descriptor)
        pointer = self.library.ladspa_descriptor(0)
        descriptor = pointer.contents
        if descriptor.label != b'deep_filter_mono':
            raise RuntimeError('Expected the mono DeepFilterNet plugin')
        instantiate = C.CFUNCTYPE(C.c_void_p, C.POINTER(Descriptor), C.c_ulong)(descriptor.instantiate)
        self.handle = instantiate(pointer, self.sample_rate)
        if not self.handle:
            raise RuntimeError('DeepFilterNet initialization failed')
        connect = C.CFUNCTYPE(None, C.c_void_p, C.c_ulong, C.POINTER(C.c_float))(descriptor.connect_port)
        self.input = np.zeros(self.frame_size, dtype=np.float32)
        self.output = np.zeros(self.frame_size, dtype=np.float32)
        self.controls = []
        defaults = {
            b'Attenuation Limit (dB)': 60., b'Min processing threshold (dB)': -15.,
            b'Max ERB processing threshold (dB)': 35.,
            b'Max DF processing threshold (dB)': 35.,
            b'Min Processing Buffer (frames)': 1., b'Post Filter Beta': 0.,
        }
        for port in range(descriptor.port_count):
            name = descriptor.port_names[port]
            if name == b'Audio In':
                data = self.input.ctypes.data_as(C.POINTER(C.c_float))
            elif name == b'Audio Out':
                data = self.output.ctypes.data_as(C.POINTER(C.c_float))
            else:
                control = C.c_float(defaults[name])
                self.controls.append(control)  # Port storage must outlive the native instance.
                data = C.pointer(control)
            connect(self.handle, port, data)
        self.run = C.CFUNCTYPE(None, C.c_void_p, C.c_ulong)(descriptor.run)
        self.deactivate = C.CFUNCTYPE(None, C.c_void_p)(descriptor.deactivate) if descriptor.deactivate else None
        self.cleanup = C.CFUNCTYPE(None, C.c_void_p)(descriptor.cleanup)
        if descriptor.activate:
            C.CFUNCTYPE(None, C.c_void_p)(descriptor.activate)(self.handle)
        # Stateful anti-alias filtering before decimation to Silero's 16 kHz.
        self.taps = firwin(63, 1 / 3).astype(np.float32)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.float32)

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape != (self.frame_size,) or not np.isfinite(audio).all():
            raise ValueError('DeepFilterNet expects 480 finite mono samples at 48 kHz')
        if not self.handle:
            raise RuntimeError('DeepFilterNet is closed')
        self.input[:] = audio
        self.run(self.handle, self.frame_size)
        filtered, self.zi = lfilter(self.taps, [1.], self.output, zi=self.zi)
        return filtered[::3].astype(np.float32)

    def close(self):
        if self.handle:
            if self.deactivate:
                self.deactivate(self.handle)
            self.cleanup(self.handle)
            self.handle = None


def denoised_frames(frames, suppressor):
    """Repack enhanced 10 ms frames into the existing 512-sample VAD frames."""
    pending = np.empty(0, dtype=np.float32)
    for frame in frames:
        pending = np.concatenate((pending, suppressor.process(frame)))
        while len(pending) >= 512:
            yield pending[:512].copy()
            pending = pending[512:]
    if len(pending):
        yield np.pad(pending, (0, 512 - len(pending)))


class BrowserNoiseSuppressor:
    """Preserve resampling and DeepFilterNet state across WebSocket packets."""
    def __init__(self, sample_rate):
        import soxr
        self.suppressor = NoiseSuppressor()
        self.resampler = None if sample_rate == 48000 else soxr.ResampleStream(
            sample_rate, 48000, 1, dtype='float32', quality='LQ')
        self.pending = np.empty(0, dtype=np.float32)

    def process(self, audio, final=False):
        audio = np.asarray(audio, dtype=np.float32)
        if self.resampler is not None:
            audio = self.resampler.resample_chunk(audio, last=final)
        self.pending = np.concatenate((self.pending, audio))
        if final:
            # Complete the last frame and flush the STFT/model lookahead.
            padding = (-len(self.pending)) % 480 + 6 * 480
            self.pending = np.pad(self.pending, (0, padding))
        output = []
        while len(self.pending) >= 480:
            output.append(self.suppressor.process(self.pending[:480]))
            self.pending = self.pending[480:]
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def close(self):
        self.suppressor.close()
