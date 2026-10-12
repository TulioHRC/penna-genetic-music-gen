"""
Groove Extractor:
Extrai o "groove" (padrão de bateria de um compasso) e o andamento de um áudio de referência.

1. Separa a parte percussiva do áudio (HPSS).
2. Detecta as batidas (beat tracking) e divide cada uma em 4 semicolcheias -> grade de 16 steps/compasso.
3. Mede o ataque (fluxo espectral) em três faixas de frequência em cada step:
   - kick:  graves (bumbo / 808)
   - snare: médios (caixa / palma)
   - hat:   agudos (chimbal / shaker)
4. Faz a média de todos os compassos e alinha o compasso para que o tempo 1 caia no step 0.

O resultado é um padrão de intensidades (0..1) por step, que o arranjador usa para tocar a
bateria e o baixo no mesmo "balanço" da música de referência.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np

STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4
PARTS = ("kick", "snare", "hat")

ANALYSIS_SR = 22050
HOP_LENGTH = 256
N_FFT = 1024
BAND_HZ = {"kick": (30, 150), "snare": (180, 2500), "hat": (6000, 11000)}

# Analisa um trecho do meio da música (intros e finais costumam ter groove diferente)
ANALYSIS_OFFSET_SECONDS = 20.0
ANALYSIS_DURATION_SECONDS = 90.0


@dataclass
class Groove:
    tempo_bpm: float
    kick: np.ndarray   # intensidade 0..1 em cada um dos 16 steps
    snare: np.ndarray
    hat: np.ndarray

    def part(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def hits(self, name: str, threshold: float) -> List[Tuple[int, float]]:
        """Steps em que a peça toca, com a intensidade de cada um."""
        return [(step, float(v)) for step, v in enumerate(self.part(name)) if v >= threshold]

    def as_text(self, threshold: float) -> Dict[str, str]:
        """Representação estilo drum machine (X forte, x fraco, . silêncio) para o log."""
        strong = (1 + threshold) / 2
        return {
            name: "".join("X" if v >= strong else "x" if v >= threshold else "." for v in self.part(name))
            for name in PARTS
        }

    @classmethod
    def from_steps(cls, tempo_bpm: float, kick: Sequence[int], snare: Sequence[int], hat: Sequence[int]) -> "Groove":
        def pattern(steps: Sequence[int]) -> np.ndarray:
            p = np.zeros(STEPS_PER_BAR)
            p[list(steps)] = 1.0
            return p
        return cls(tempo_bpm, pattern(kick), pattern(snare), pattern(hat))


def _load_middle_excerpt(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=ANALYSIS_SR, mono=True)
    start = int(ANALYSIS_OFFSET_SECONDS * ANALYSIS_SR)
    end = start + int(ANALYSIS_DURATION_SECONDS * ANALYSIS_SR)
    # Música curta: usa o áudio inteiro
    return y[start:end] if len(y) > end else y


def _sixteenth_grid(beat_times: np.ndarray) -> np.ndarray:
    """Divide cada intervalo entre batidas em 4 semicolcheias."""
    grid = [np.linspace(a, b, STEPS_PER_BEAT + 1)[:-1] for a, b in zip(beat_times[:-1], beat_times[1:])]
    return np.concatenate(grid) if grid else np.array([])


def _band_onsets(spectrum: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Fluxo espectral (aumento de energia) na faixa de frequência: alto onde há um ataque."""
    lo, hi = band
    energy = np.log1p(spectrum[(freqs >= lo) & (freqs < hi)].sum(axis=0))
    return np.maximum(0.0, np.diff(energy, prepend=energy[0]))


def _contrast(values: np.ndarray) -> np.ndarray:
    """Normaliza para 0..1 destacando os picos sobre o 'ruído de fundo' do compasso."""
    floor = np.median(values)
    return np.clip((values - floor) / (values.max() - floor + 1e-9), 0.0, 1.0)


def _align_downbeat(parts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    O beat tracker não sabe onde o compasso começa (e às vezes trava no contratempo).
    Testa rotações de colcheia em colcheia e escolhe a que põe o bumbo forte no step 0
    e a caixa fora dele, que é o caso na grande maioria dos gêneros populares.
    """
    def score(rotation: int) -> float:
        return np.roll(parts["kick"], -rotation)[0] - np.roll(parts["snare"], -rotation)[0]

    best = max(range(0, STEPS_PER_BAR, 2), key=score)
    return {name: np.roll(p, -best) for name, p in parts.items()}


def extract_groove(audio_path: str) -> Groove:
    y = _load_middle_excerpt(audio_path)
    _, percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=percussive, sr=ANALYSIS_SR, hop_length=HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR, hop_length=HOP_LENGTH)
    grid_frames = librosa.time_to_frames(_sixteenth_grid(beat_times), sr=ANALYSIS_SR, hop_length=HOP_LENGTH)

    spectrum = np.abs(librosa.stft(percussive, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=ANALYSIS_SR, n_fft=N_FFT)
    full_bars = len(grid_frames) // STEPS_PER_BAR * STEPS_PER_BAR
    if full_bars == 0:
        raise ValueError(f"Áudio curto demais para extrair groove: {audio_path}")

    parts = {}
    for name in PARTS:
        onsets = _band_onsets(spectrum, freqs, BAND_HZ[name])
        # Pico numa janela de ±1 frame absorve pequenas imprecisões da grade
        per_step = np.array([onsets[max(0, f - 1): f + 2].max() for f in grid_frames[:full_bars]])
        parts[name] = _contrast(per_step.reshape(-1, STEPS_PER_BAR).mean(axis=0))

    parts = _align_downbeat(parts)
    return Groove(float(np.atleast_1d(tempo)[0]), parts["kick"], parts["snare"], parts["hat"])


def average_grooves(grooves: Sequence[Groove]) -> Groove:
    return Groove(
        tempo_bpm=float(np.median([g.tempo_bpm for g in grooves])),
        **{name: np.mean([g.part(name) for g in grooves], axis=0) for name in PARTS},
    )
