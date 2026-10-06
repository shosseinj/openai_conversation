"""Independent frontend A/B/C/D/E diagnostic; never imports speaker verification.

python diagnose_pipeline.py recording.wav --reference-file correct_transcript.txt
Production files are untouched. D and E intentionally have identical audio paths.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import time
import unicodedata

import numpy as np
import soundfile as sf
import torch
from live_vad import SpeechDetector
from noise_suppression import BrowserNoiseSuppressor
from transcribe import load_transcriber

RATE = 16000
PACKET = 4096


def decode(path, rate):
    result = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
         '-f', 'f32le', '-ac', '1', '-ar', str(rate), 'pipe:1'],
        capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Input must contain nonempty, finite audio')
    return audio


def levels(audio):
    x = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    clipped = int(np.count_nonzero(np.abs(x) >= .999))
    return dict(sample_rate=RATE, channels=1, dtype=str(x.dtype),
                duration=len(x) / RATE, samples=len(x), rms=rms,
                dbfs=20 * np.log10(rms) if rms else None,
                peak=float(np.max(np.abs(x))) if x.size else 0.0,
                near_full_scale_samples=clipped,
                near_full_scale_percent=100 * clipped / len(x) if x.size else 0.0)


def normalize(text):
    """One normalization for reference and ALL hypotheses; ZWNJ is a word boundary."""
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans('يك', 'یک'))
    result = []
    for c in text:
        category = unicodedata.category(c)
        if category.startswith('M') or c == '\u0640':
            continue  # Diacritics and tatweel.
        if c == '\u200c' or category.startswith('P'):
            result.append(' ')
        elif category == 'Cf':
            continue  # Bidi/format controls.
        else:
            result.append(c)
    return ' '.join(''.join(result).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def errors(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError('Normalized reference must not be empty')
    return dict(cer=distance(ref, hyp) / len(ref),
                wer=distance(ref.split(), hyp.split()) / len(ref.split()),
                normalized_reference=ref, normalized_transcript=hyp)


@contextmanager
def capture_native_stderr(path):
    """Capture the native LADSPA logger, which bypasses Python logging."""
    saved = os.dup(2)
    try:
        with open(path, 'wb') as output:
            os.dup2(output.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def denoise(audio48, logfile):
    chunks = []
    with capture_native_stderr(logfile):
        suppressor = BrowserNoiseSuppressor(48000)
        try:
            started = time.perf_counter()
            # Keep the existing native implementation and its ordinary stop flush.
            # Pace packets to avoid feeding a real-time plugin as an offline burst.
            for offset in range(0, len(audio48), PACKET):
                deadline = started + offset / 48000
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                chunks.append(suppressor.process(audio48[offset:offset + PACKET]))
            chunks.append(suppressor.process(np.empty(0, dtype=np.float32), True))
        finally:
            suppressor.close()
    return chunks


def frontend_segments(chunks, detector=None):
    """Mirror app.listen.accept_audio/update(final=True), including packet boundaries.

    No offline get_speech_timestamps defaults, tail trimming or new VAD tuning.
    Final explicit stop flushes buffered speech, just as the frontend does.
    """
    detector = detector if detector is not None else SpeechDetector(RATE)
    pending = np.empty(0, dtype=np.float32)
    has_speech = False
    consumed = 0
    segments = []
    def commit():
        nonlocal pending, has_speech
        if has_speech:
            segments.append((consumed - len(pending), consumed, pending.copy()))
        pending = np.empty(0, dtype=np.float32)
        has_speech = False
    for chunk in chunks:
        if not len(chunk):
            continue
        chunk = np.asarray(chunk, dtype=np.float32)
        consumed += len(chunk)
        if detector.feed(chunk):
            has_speech = True
        pending = np.concatenate((pending, chunk))
        if not has_speech:
            pending = pending[-RATE // 3:]
            continue
        if detector.silent_samples >= RATE * .75 or len(pending) >= RATE * 20:
            commit()
    commit()
    return segments


def run_test(letter, label, segments, model, output, timestamps, reference, input_samples, vad):
    prefix = {'A': 'A_original', 'B': 'B_deepfilter', 'C': 'C_vad',
              'D': 'D_deepfilter_vad', 'E': 'E_without_speaker'}[letter]
    records, texts = [], []
    for i, (start, end, audio) in enumerate(segments, 1):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        filename = f'{prefix}.wav' if len(segments) == 1 else f'{prefix}_{i:03d}.wav'
        # FLOAT WAV preserves the waveform passed to Whisper without PCM16 quantization.
        sf.write(output / filename, audio, RATE, subtype='FLOAT')
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model({'array': audio, 'sampling_rate': RATE},
                       return_timestamps=timestamps,
                       generate_kwargs={'language': 'persian', 'task': 'transcribe'})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        text = result['text'].strip()
        texts.append(text)
        records.append(dict(file=filename, start_sample=start, end_sample=end,
                            transcript=text, whisper_time=elapsed, **levels(audio)))
    transcript = ' '.join(texts)
    total = sum(r['duration'] for r in records)
    stats = levels(np.concatenate([x[2] for x in segments]) if segments else np.empty(0, dtype=np.float32))
    row = dict(test=letter, pipeline=label, transcript=transcript,
               audio_duration=total, whisper_time=sum(r['whisper_time'] for r in records),
               vad_segments=len(segments) if vad else None,
               duration_removed_by_vad=(input_samples - sum(len(x[2]) for x in segments)) / RATE if vad else 0,
               input_duration=input_samples / RATE, statistics=stats, segments=records)
    if reference is not None:
        row.update(errors(reference, transcript))
    print(f'\nTEST {letter}: {label}\nTranscription: {transcript or "[no speech segments]"}\n'
          f'Audio duration: {total:.3f}s\nWhisper processing time: {row["whisper_time"]:.3f}s', flush=True)
    print('Statistics:', json.dumps(stats, ensure_ascii=False))
    print(f'VAD segments: {row["vad_segments"] if vad else "not applied"}; '
          f'duration removed by VAD: {row["duration_removed_by_vad"]:.3f}s', flush=True)
    if reference is not None:
        print(f'CER: {100 * row["cer"]:.2f}% | WER: {100 * row["wer"]:.2f}%', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--reference', help='Correct Persian transcript')
    group.add_argument('--reference-file', type=Path, help='UTF-8 correct transcript')
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error('Audio file does not exist')
    if not torch.cuda.is_available():
        parser.error('CUDA is required; this benchmark never silently falls back to CPU')
    reference = args.reference_file.read_text(encoding='utf-8') if args.reference_file else args.reference
    if reference is not None and not normalize(reference):
        parser.error('Reference is empty after normalization')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Avoid mixing old segment files with a new benchmark.
    if any(output.glob('[ABCDE]_*.wav')) or (output / 'report.json').exists():
        parser.error('Output contains previous results; choose a fresh --output-dir')
    info = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', str(args.audio)], capture_output=True, text=True, check=True)
    print('Input metadata:', info.stdout, flush=True)
    original = decode(args.audio, RATE)
    audio48 = decode(args.audio, 48000)
    print('Original decoded statistics:', json.dumps(levels(original)), flush=True)
    cleaned_chunks = denoise(audio48, output / 'deepfilter.log')
    cleaned = np.concatenate(cleaned_chunks)
    log = (output / 'deepfilter.log').read_text(errors='replace')
    warnings = [line for line in log.splitlines() if 'WARN' in line or 'ERROR' in line]
    underruns = log.count('Underrun detected')
    print(f'DeepFilterNet underruns: {underruns}; warning/error lines: {len(warnings)}', flush=True)
    # C uses the SAME packet lengths/stop-tail as D, without its waveform processing.
    # Sample-index mapping follows the existing 480-sample DF frame repacking.
    plain_chunks, offset = [], 0
    for chunk in cleaned_chunks:
        count = len(chunk)
        plain_chunks.append(np.pad(original[offset:offset + count],
                                   (0, max(0, count - len(original[offset:offset + count])))))
        offset += count
    assert offset >= len(original), 'Packet mapping lost input samples'
    c_segments = frontend_segments(plain_chunks)
    d_segments = frontend_segments(cleaned_chunks)
    print('Loading the existing Whisper once...', flush=True)
    model = load_transcriber()
    if model.model.device.type != 'cuda' or model.model.dtype != torch.float16:
        raise RuntimeError('Whisper must run on CUDA with FP16')
    # Long-form inputs require timestamps. Choose once, identically for all branches.
    timestamps = max(len(original), len(cleaned)) > RATE * 30
    config = dict(model_path=str(Path(__file__).resolve().parent / 'model'),
                  device=str(model.model.device), dtype=str(model.model.dtype),
                  generation=model.generation_config.to_dict(),
                  call=dict(language='persian', task='transcribe', return_timestamps=timestamps),
                  vad=dict(onset=.5, continuing=.35, silence_seconds=.75,
                           preroll_samples=-(-RATE // 3), maximum_seconds=20),
                  speaker_verification=False, deepfilter_shared_across_B_D_E=True)
    print(f'Whisper: {model.model.device}, {model.model.dtype}; input Float32 mono 16000 Hz; Persian/transcribe; timestamps={timestamps}', flush=True)
    jobs = [('A', 'Original → Whisper', [(0, len(original), original)], len(original), False),
            ('B', 'DeepFilterNet → Whisper', [(0, len(cleaned), cleaned)], len(cleaned), False),
            ('C', 'VAD → Whisper', c_segments, offset, True),
            ('D', 'DeepFilterNet → VAD → Whisper', d_segments, len(cleaned), True),
            ('E', 'DeepFilterNet → VAD → Whisper (speaker bypass)', d_segments, len(cleaned), True)]
    print('D/E are identical by specification; E repeats decoding the same samples, never speaker filtering.', flush=True)
    rows = []
    for letter, label, segments, samples, vad in jobs:
        row = run_test(letter, label, segments, model, output, timestamps, reference, samples, vad)
        row["deepfilter_underruns"] = underruns if letter in "BDE" else 0
        row["deepfilter_warning_lines"] = len(warnings) if letter in "BDE" else 0
        print(f'DeepFilterNet underruns: {row["deepfilter_underruns"]}; warning/error lines: {row["deepfilter_warning_lines"]}', flush=True)
        rows.append(row)
        report = dict(input=str(args.audio.resolve()), input_metadata=json.loads(info.stdout),
                      configuration=config, deepfilter_underruns=underruns,
                      deepfilter_warnings=warnings, results=rows,
                      notes=['No browser DSP, WebSocket backlog, or speaker rejection simulated.',
                             'C/D reproduce frontend packet segmentation, including explicit stop.',
                             'DF retains production stop flush and native timing artifacts.',
                             'C includes zero stop-tail padding to match D packet lengths.',
                             'Clipping metric counts near-full-scale samples, not proven analog clipping.',
                             'A includes first-call warmup; times are observed, not an optimized latency benchmark.',
                             'CER includes normalized spaces; WER uses whitespace tokens.'])
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('\nTest | Pipeline | Transcript | Audio Duration | Whisper Time' + (' | CER | WER' if reference is not None else ''))
    for r in rows:
        text = r['transcript'].replace('|', '¦').replace('\n', ' ')
        print(f'{r["test"]} | {r["pipeline"]} | {text} | {r["audio_duration"]:.3f}s | {r["whisper_time"]:.3f}s' +
              (f' | {100*r["cer"]:.2f}% | {100*r["wer"]:.2f}%' if reference is not None else ''))
    print('\nDiagnostic interpretation (conditional, not a proven cause):\n'
          'A bad: microphone/audio or Whisper/model.\n'
          'A good, B worse: DeepFilterNet/resampling/runtime path.\n'
          'A good, C worse: VAD/segmentation.\n'
          'B and C good, D worse: DeepFilterNet/VAD interaction.\n'
          'A–D good, live app bad: browser/streaming/buffering/speaker verification.')
    if reference is None:
        print('No ground truth supplied: accuracy and the responsible stage cannot be established from transcripts alone.')
    else:
        best = min(rows, key=lambda r: (r['wer'], r['cer']))
        print(f'Lowest measured error: test {best["test"]}. Review WAV boundaries and pairwise error changes before attributing cause.')
    print(f'Results saved to {output}', flush=True)


if __name__ == '__main__':
    main()
