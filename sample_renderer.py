"""
Sample Renderer:
Toca o arranjo MIDI com os sons recortados da música de referência (sample_kit.py).
- Bateria: bumbo, caixa/palma e chimbal viram os samples da referência.
- Baixo e lead: o sample de nota é transposto para cada nota (reamostragem).
- O que não tem sample (pratos, viradas, acordes) é tocado pela SoundFont.

Cada pista é nivelada por RMS e mixada com ganhos fixos, já que os samples
vêm com volumes bem diferentes entre si.
"""

from typing import Dict, Optional

import numpy as np
import pretty_midi
import scipy.signal as signal

from sample_kit import SampleKit
from soundfont_renderer import TAIL_SECONDS, master, render_soundfont_dry

DRUM_PITCH_TO_PART = {35: "kick", 36: "kick", 38: "snare", 39: "snare", 40: "snare", 42: "hat", 44: "hat"}
TRACK_GAINS = {"Drums": 1.0, "Bass": 0.75, "Lead": 0.6, "SoundFont": 0.45}
VELOCITY_CURVE = 1.6  # >1: velocity baixa fica bem mais suave, como num instrumento real
RELEASE_SECONDS = 0.04
# Transpor um sample gravado muito longe da nota original soa artificial (lento/"esquilo");
# notas mais distantes que isso são trazidas para a oitava mais próxima (mesma nota, outra oitava)
MAX_TRANSPOSE_SEMITONES = 6


def _velocity_gain(velocity: int) -> float:
    return (velocity / 127.0) ** VELOCITY_CURVE


def _transpose(sample: np.ndarray, semitones: float) -> np.ndarray:
    """
    Muda a altura por reamostragem (como num sampler clássico: mais agudo = mais curto).
    Reamostragem por FFT é limitada em banda: transpor para cima não gera aliasing
    (a interpolação linear gerava chiado agudo).
    """
    if semitones == 0:
        return sample
    new_length = max(1, int(round(len(sample) / 2.0 ** (semitones / 12.0))))
    return signal.resample(sample, new_length)


def _semitones_from_root(pitch: int, root_pitch: int) -> int:
    semitones = pitch - root_pitch
    while semitones > MAX_TRANSPOSE_SEMITONES:
        semitones -= 12
    while semitones < -MAX_TRANSPOSE_SEMITONES:
        semitones += 12
    return semitones


def _place(track: np.ndarray, audio: np.ndarray, start_sample: int) -> None:
    end = min(len(track), start_sample + len(audio))
    if end > start_sample:
        track[start_sample:end] += audio[: end - start_sample]


def _render_drums(instrument: pretty_midi.Instrument, kit: SampleKit, track: np.ndarray, sr: int) -> list:
    """Coloca os samples da bateria; devolve as notas sem sample (pratos, tons) para a SoundFont."""
    leftovers = []
    for note in instrument.notes:
        part = DRUM_PITCH_TO_PART.get(note.pitch)
        if part in kit.drums:
            _place(track, kit.drums[part] * _velocity_gain(note.velocity), int(note.start * sr))
        else:
            leftovers.append(note)
    return leftovers


def _render_pitched(instrument: pretty_midi.Instrument, sample, track: np.ndarray, sr: int) -> None:
    release = int(RELEASE_SECONDS * sr)
    transposed = {}
    for note in instrument.notes:
        semitones = _semitones_from_root(note.pitch, sample.root_pitch)
        if semitones not in transposed:
            transposed[semitones] = _transpose(sample.audio, semitones)
        audio = transposed[semitones].copy()
        length = min(len(audio), int((note.end - note.start) * sr) + release)
        audio = audio[:length] * _velocity_gain(note.velocity)
        fade = min(release, length)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade)
        _place(track, audio, int(note.start * sr))


def _rms_normalized(track: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(track ** 2))
    return track / rms if rms > 0 else track


def render_with_samples(
    pm: pretty_midi.PrettyMIDI, kit: SampleKit, soundfont_path: Optional[str], sr: int
) -> np.ndarray:
    total_samples = int((pm.get_end_time() + TAIL_SECONDS) * sr)
    tracks: Dict[str, np.ndarray] = {}
    soundfont_pm = pretty_midi.PrettyMIDI(initial_tempo=pm.get_tempo_changes()[1][0])

    for instrument in pm.instruments:
        track = np.zeros(total_samples)
        pitched_sample = kit.pitched.get(instrument.name.lower())
        if instrument.is_drum and kit.drums:
            leftovers = _render_drums(instrument, kit, track, sr)
            if leftovers:
                rest = pretty_midi.Instrument(instrument.program, is_drum=True, name="Drums (SoundFont)")
                rest.notes = leftovers
                soundfont_pm.instruments.append(rest)
        elif pitched_sample is not None:
            _render_pitched(instrument, pitched_sample, track, sr)
        else:
            soundfont_pm.instruments.append(instrument)
            continue
        tracks[instrument.name] = track

    mix = np.zeros((total_samples, 2))
    for name, track in tracks.items():
        mix += (TRACK_GAINS.get(name, 0.6) * 0.1 * _rms_normalized(track))[:, None]

    if soundfont_pm.instruments and soundfont_path:
        rendered = render_soundfont_dry(soundfont_pm, soundfont_path, sr)[:total_samples]
        mix[: len(rendered)] += TRACK_GAINS["SoundFont"] * 0.1 * _rms_normalized(rendered)

    return master(mix.astype(np.float32), sr)
