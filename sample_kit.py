"""
Sample Kit Extractor:
Recorta da própria música de referência os sons usados para tocar o arranjo gerado,
para que o resultado tenha o timbre da referência (e não o de instrumentos General MIDI).

Sem separação de fontes por rede neural (lenta em CPU): usa HPSS do librosa.
- Bateria (kick / snare / hat): ataques na parte percussiva, classificados pela faixa de
  frequência dominante. Os N ataques mais típicos de cada peça são alinhados e somados:
  como a mesma amostra de bateria se repete na música, ela se reforça, enquanto voz e
  instrumentos que vazam por cima (diferentes a cada batida) se cancelam.
- Baixo: o trecho de nota mais estável e forte da parte harmônica grave, guardado junto
  com a nota MIDI dele para ser transposto.
- Lead: mesmo critério, mas no canal lateral (L-R) em vez do mono. A voz é mixada no centro
  e se cancela em L-R, sobrando só instrumentos abertos no estéreo (guitarras, synths).
  Se a música não tem conteúdo estéreo suficiente, o lead fica sem sample (SoundFont toca),
  para nunca usar a voz como instrumento.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf

KIT_SR = 44100
EXCERPT_OFFSET_SECONDS = 20.0
EXCERPT_DURATION_SECONDS = 60.0
HPSS_MARGIN = 2.0  # separa mais agressivamente percussão de harmonia

# Bateria
DRUM_BANDS_HZ = {"kick": (30, 150), "snare": (200, 2500), "hat": (6000, 14000)}
DRUM_SAMPLE_SECONDS = {"kick": 0.40, "snare": 0.30, "hat": 0.12}
ATTACK_WINDOW_SECONDS = 0.04
# Faixa útil de cada peça: o que fica fora dela é vazamento de outros instrumentos
DRUM_CLEANUP_HZ = {"kick": (25, 2000), "snare": (120, 12000), "hat": (5000, 18000)}
HITS_TO_AVERAGE = 12
ALIGN_MAX_SHIFT_SECONDS = 0.005
PRE_ONSET_SECONDS = 0.005

# Notas (baixo / lead)
PITCHED_BANDS_HZ = {"bass": (30, 300), "lead": (250, 4000)}
# Energia mínima do canal lateral (relativa ao centro) para ter instrumentos fora do centro
MIN_SIDE_TO_MID_RATIO = 0.08
PITCH_RANGES = {"bass": ("E1", "G3"), "lead": ("C3", "C6")}
MIN_STABLE_NOTE_SECONDS = {"bass": 0.25, "lead": 0.30}
MAX_PITCHED_SAMPLE_SECONDS = 1.0
PITCH_HOP = 512
PITCH_FRAME_LENGTH = 4096  # janela longa o bastante para graves de baixo (E1 ≈ 41 Hz)
MAX_PITCH_WOBBLE_SEMITONES = 0.35

FADE_OUT_SECONDS = 0.02
# Fade-in curto: o recorte começa no meio da onda e sem ele cada nota começa com um clique.
# Na bateria é mais curto para não amaciar o ataque.
FADE_IN_SECONDS = {"drum": 0.001, "pitched": 0.005}

# Decaimento natural da bateria: mede quanto tempo a batida leva para cair à metade logo após
# o pico e impede o resto do sample de ficar acima dessa curva exponencial. O que "persiste"
# depois da batida (voz/instrumentos que vazaram) é cortado; a cauda real do tambor fica.
ENVELOPE_SMOOTHING_SECONDS = 0.005
# Medido até 25% do pico (e não 50%): o clique inicial do bumbo cai muito rápido e
# subestimaria a cauda. Meia-vida mínima por peça preserva a cauda grave de um 808.
DECAY_REFERENCE_LEVEL = 0.25
MIN_HALF_LIFE_SECONDS = {"kick": 0.08, "snare": 0.04, "hat": 0.015}


@dataclass
class PitchedSample:
    audio: np.ndarray
    root_pitch: int  # nota MIDI gravada no sample


@dataclass
class SampleKit:
    drums: Dict[str, np.ndarray] = field(default_factory=dict)
    pitched: Dict[str, PitchedSample] = field(default_factory=dict)
    sr: int = KIT_SR

    def describe(self) -> str:
        parts = list(self.drums) + [
            f"{name} ({librosa.midi_to_note(s.root_pitch)})" for name, s in self.pitched.items()
        ]
        return ", ".join(parts) if parts else "nenhum som extraído"

    def save(self, directory: str) -> None:
        """Salva cada sample em .wav, para ouvir o que foi extraído."""
        os.makedirs(directory, exist_ok=True)
        for old in os.listdir(directory):  # remove samples de execuções anteriores
            if old.endswith(".wav"):
                os.remove(os.path.join(directory, old))
        for name, audio in self.drums.items():
            sf.write(os.path.join(directory, f"{name}.wav"), audio, self.sr)
        for name, sample in self.pitched.items():
            note = librosa.midi_to_note(sample.root_pitch).replace("♯", "#")
            sf.write(os.path.join(directory, f"{name}_{note}.wav"), sample.audio, self.sr)


# ---------- utilitários ----------

def _bandpass(y: np.ndarray, band: Tuple[float, float], sr: int) -> np.ndarray:
    lo, hi = band
    sos = signal.butter(4, [lo, min(hi, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y)


def _finish(sample: np.ndarray, sr: int, kind: str) -> np.ndarray:
    """Fade-in e fade-out curtos (sem cliques) e normalização de pico."""
    sample = sample.copy()
    fade_in = min(len(sample), max(1, int(FADE_IN_SECONDS[kind] * sr)))
    fade_out = min(len(sample), int(FADE_OUT_SECONDS * sr))
    sample[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    sample[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    return sample / (np.max(np.abs(sample)) + 1e-9)


def _smooth(values: np.ndarray, sr: int) -> np.ndarray:
    window = max(1, int(ENVELOPE_SMOOTHING_SECONDS * sr))
    return np.convolve(values, np.ones(window) / window, mode="same")


def _enforce_natural_decay(sample: np.ndarray, sr: int, part: str) -> np.ndarray:
    """Limita o envelope, após o pico, a um decaimento exponencial medido na própria batida."""
    envelope = _smooth(np.abs(sample), sr)
    peak = int(np.argmax(envelope))
    below_reference = np.nonzero(envelope[peak:] < DECAY_REFERENCE_LEVEL * envelope[peak])[0]
    # Chegar a 25% leva duas meias-vidas
    measured = below_reference[0] / 2 if len(below_reference) else len(sample) - peak
    half_life = max(MIN_HALF_LIFE_SECONDS[part] * sr, measured)

    t = np.arange(len(sample) - peak)
    allowed = envelope[peak] * 0.5 ** (t / half_life)
    gain = np.ones(len(sample))
    gain[peak:] = np.minimum(1.0, allowed / (envelope[peak:] + 1e-12))
    return sample * _smooth(gain, sr)  # ganho suavizado: sem "chiado" de modulação


def _load_mid_side(audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Trecho do meio da música como (centro, lateral). Áudio mono tem lateral zerado."""
    y, _ = librosa.load(audio_path, sr=KIT_SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    start = int(EXCERPT_OFFSET_SECONDS * KIT_SR)
    end = start + int(EXCERPT_DURATION_SECONDS * KIT_SR)
    if y.shape[1] > end:
        y = y[:, start:end]
    left, right = y[0], y[1]
    return (left + right) / 2, (left - right) / 2


# ---------- bateria ----------

def _band_share_at_onsets(percussive: np.ndarray, onsets: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Fração da energia do ataque que cai em cada faixa (kick/snare/hat)."""
    window = int(ATTACK_WINDOW_SECONDS * sr)
    energy = {}
    for name, band in DRUM_BANDS_HZ.items():
        filtered = _bandpass(percussive, band, sr)
        energy[name] = np.array([np.sum(filtered[o:o + window] ** 2) for o in onsets])
    total = sum(energy.values()) + 1e-12
    return {name: e / total for name, e in energy.items()}


def _align_to(reference: np.ndarray, clip: np.ndarray, max_shift: int) -> np.ndarray:
    """Desloca `clip` (até ±max_shift amostras) para coincidir com `reference`."""
    correlation = signal.correlate(clip, reference, mode="full", method="fft")
    center = len(reference) - 1
    window = correlation[center - max_shift: center + max_shift + 1]
    shift = int(np.argmax(window)) - max_shift
    # Desloca preenchendo com zeros (np.roll jogaria o fim do recorte para o começo)
    aligned = np.zeros_like(clip)
    if shift >= 0:
        aligned[: len(clip) - shift] = clip[shift:]
    else:
        aligned[-shift:] = clip[: len(clip) + shift]
    return aligned


def _average_hits(percussive: np.ndarray, onsets: np.ndarray, length: int, sr: int) -> np.ndarray:
    pre = int(PRE_ONSET_SECONDS * sr)
    clips = [percussive[o - pre: o - pre + length] for o in onsets if o >= pre and o - pre + length <= len(percussive)]
    reference = clips[0]
    max_shift = int(ALIGN_MAX_SHIFT_SECONDS * sr)
    aligned = [reference] + [_align_to(reference, c, max_shift) for c in clips[1:]]
    return np.mean(aligned, axis=0)


def extract_drum_samples(percussive: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    onset_frames = librosa.onset.onset_detect(y=percussive, sr=sr, backtrack=True, units="samples")
    if len(onset_frames) < 3:
        return {}
    loudness = np.array([np.max(np.abs(percussive[o:o + int(ATTACK_WINDOW_SECONDS * sr)])) for o in onset_frames])
    shares = _band_share_at_onsets(percussive, onset_frames, sr)

    drums = {}
    for name in DRUM_BANDS_HZ:
        # Ataques em que esta peça domina e que não são fracos demais
        typical = (shares[name] >= np.percentile(shares[name], 80)) & (loudness >= np.median(loudness))
        candidates = onset_frames[typical]
        if len(candidates) == 0:
            continue
        best = candidates[np.argsort(-shares[name][typical])[:HITS_TO_AVERAGE]]
        length = int(DRUM_SAMPLE_SECONDS[name] * sr)
        averaged = _average_hits(percussive, np.sort(best), length, sr)
        drums[name] = _finish(_enforce_natural_decay(_bandpass(averaged, DRUM_CLEANUP_HZ[name], sr), sr, name), sr, "drum")
    return drums


# ---------- baixo / lead ----------

def _most_stable_note(y: np.ndarray, sr: int, part: str) -> Optional[PitchedSample]:
    """Trecho mais forte em que a altura fica estável, e a nota MIDI dele."""
    fmin, fmax = (librosa.note_to_hz(n) for n in PITCH_RANGES[part])
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=PITCH_FRAME_LENGTH, hop_length=PITCH_HOP)
    midi = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(y=y, hop_length=PITCH_HOP)[0][: len(midi)]
    voiced = rms > np.percentile(rms, 60)

    min_frames = int(MIN_STABLE_NOTE_SECONDS[part] * sr / PITCH_HOP)
    best: Optional[Tuple[float, int, int]] = None  # (energia, início, fim) em frames
    start = 0
    for i in range(1, len(midi) + 1):
        segment_breaks = (
            i == len(midi) or not voiced[i]
            or abs(midi[i] - np.median(midi[start:i])) > MAX_PITCH_WOBBLE_SEMITONES
        )
        if segment_breaks:
            if i - start >= min_frames and voiced[start]:
                energy = float(np.mean(rms[start:i]))
                if best is None or energy > best[0]:
                    best = (energy, start, i)
            start = i
    if best is None:
        return None

    _, first, last = best
    begin = first * PITCH_HOP
    end = min(last * PITCH_HOP, begin + int(MAX_PITCHED_SAMPLE_SECONDS * sr))
    root = int(round(np.median(midi[first:last])))
    return PitchedSample(audio=_finish(y[begin:end], sr, "pitched"), root_pitch=root)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)))


def extract_sample_kit(audio_path: str) -> SampleKit:
    mid, side = _load_mid_side(audio_path)
    harmonic, percussive = librosa.effects.hpss(mid, margin=HPSS_MARGIN)
    kit = SampleKit(drums=extract_drum_samples(percussive, KIT_SR))

    bass = _most_stable_note(_bandpass(harmonic, PITCHED_BANDS_HZ["bass"], KIT_SR), KIT_SR, "bass")
    if bass is not None:
        kit.pitched["bass"] = bass

    if _rms(side) >= MIN_SIDE_TO_MID_RATIO * _rms(mid):
        side_harmonic, _ = librosa.effects.hpss(side, margin=HPSS_MARGIN)
        lead = _most_stable_note(_bandpass(side_harmonic, PITCHED_BANDS_HZ["lead"], KIT_SR), KIT_SR, "lead")
        if lead is not None:
            kit.pitched["lead"] = lead
    return kit
