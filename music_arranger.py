"""
Arranger and Multi-Track MIDI Generator:
Transforma a melodia evoluída pelo Algoritmo Genético em um arranjo de banda
que *segue a melodia* em vez de aplicar um template fixo:

1. Detecta a tonalidade (tônica + modo) a partir do perfil do corpus (Krumhansl-Schmuckler).
2. Harmoniza cada compasso escolhendo o acorde diatônico que melhor contém as notas da melodia.
3. Gera as pistas de acompanhamento a partir dessa progressão, no estilo escolhido (styles.py):
   - Lead: a melodia evoluída, com o timbre do estilo (guitarra no rock, synth no funk).
   - Harmonia: acordes sustentados por compasso (só em estilos que usam acordes).
   - Baixo: fundamental do acorde, tocando junto com o bumbo do groove.
   - Bateria: o groove extraído da referência (groove_extractor.py), com o kit do estilo.
4. Humaniza todas as pistas (pequenas variações de tempo e intensidade), como um músico tocando.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi

from alg_genetic_model import Individual
from groove_extractor import Groove
from styles import CLOSED_HAT, KICK, StylePreset

STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16
BARS_PER_PHRASE = 4

# Perfis de Krumhansl-Kessler (índice 0 = tônica)
MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Tríades diatônicas por grau: (offset da fundamental em relação à tônica, intervalos da tríade)
MAJOR_TRIAD = (0, 4, 7)
MINOR_TRIAD = (0, 3, 7)
DIMINISHED_TRIAD = (0, 3, 6)
DIATONIC_CHORDS = {
    "major": [(0, MAJOR_TRIAD), (2, MINOR_TRIAD), (4, MINOR_TRIAD), (5, MAJOR_TRIAD),
              (7, MAJOR_TRIAD), (9, MINOR_TRIAD)],
    # Menor natural + V maior (dominante da menor harmônica, típica do rock/pop)
    "minor": [(0, MINOR_TRIAD), (3, MAJOR_TRIAD), (5, MINOR_TRIAD), (7, MINOR_TRIAD),
              (7, MAJOR_TRIAD), (8, MAJOR_TRIAD), (10, MAJOR_TRIAD)],
}

# Bônus para acordes de função forte (I, IV, V) e para manter o acorde anterior
FUNCTIONAL_DEGREES = {0, 5, 7}
FUNCTIONAL_BONUS = 0.15
CHORD_CONTINUITY_BONUS = 0.10

# GM Drum Map (viradas e pratos)
CRASH, LOW_TOM, MID_TOM, HIGH_TOM = 49, 45, 47, 50
FILL_PITCHES = (38, 38, HIGH_TOM, HIGH_TOM, MID_TOM, MID_TOM, LOW_TOM, LOW_TOM)

# Groove: intensidade mínima (0..1) para um step virar uma batida, e a faixa de velocity resultante
GROOVE_HIT_THRESHOLD = 0.45
DRUM_VELOCITY_MIN, DRUM_VELOCITY_MAX = 60, 120
BASS_VELOCITY = 100

# Humanização: desvios aleatórios pequenos o bastante para soar "tocado", não "errado"
TIMING_JITTER_SECONDS = 0.010
DRUM_TIMING_JITTER_SECONDS = 0.004  # bateria fica mais firme: ela é a referência de tempo
VELOCITY_JITTER = 7
DOWNBEAT_ACCENT = 8
MIN_NOTE_SECONDS = 0.03


@dataclass(frozen=True)
class Key:
    tonic_pc: int
    mode: str  # "major" | "minor"


@dataclass(frozen=True)
class Chord:
    root_pc: int
    intervals: Tuple[int, ...]

    @property
    def pitch_classes(self) -> List[int]:
        return [(self.root_pc + i) % 12 for i in self.intervals]


def detect_key(pitch_class_distribution: Sequence[float]) -> Key:
    """Estima a tonalidade correlacionando o histograma de pitch classes com os perfis K-K."""
    distribution = np.asarray(pitch_class_distribution, dtype=float)
    best_key, best_score = Key(0, "major"), -np.inf
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_KEY_PROFILE), ("minor", MINOR_KEY_PROFILE)):
            score = np.corrcoef(distribution, np.roll(profile, tonic))[0, 1]
            if score > best_score:
                best_key, best_score = Key(tonic, mode), score
    return best_key


def _bar_pitch_class_weights(individual: Individual, total_bars: int) -> List[np.ndarray]:
    """Peso de cada pitch class por compasso (duração soando, com ênfase em tempos fortes)."""
    weights = [np.zeros(12) for _ in range(total_bars)]
    step = 0
    for gene in individual.genes:
        if not gene.is_rest:
            for s in range(step, step + gene.duration_steps):
                bar = s // STEPS_PER_BAR
                if bar >= total_bars:
                    break
                strong_beat = (s % STEPS_PER_BEAT) == 0
                weights[bar][gene.pitch % 12] += 2.0 if strong_beat else 1.0
        step += gene.duration_steps
    return weights


def _chord_score(degree: int, chord: Chord, pc_share: np.ndarray, previous: Optional[Chord]) -> float:
    """Fração da melodia coberta pelo acorde, com bônus para função tonal forte e continuidade."""
    coverage = sum(pc_share[pc] for pc in chord.pitch_classes)
    bonus = FUNCTIONAL_BONUS if degree in FUNCTIONAL_DEGREES else 0.0
    if chord == previous:
        bonus += CHORD_CONTINUITY_BONUS
    return coverage + bonus


def harmonize(individual: Individual, key: Key, total_bars: int) -> List[Chord]:
    """Escolhe um acorde diatônico por compasso que melhor acomoda as notas da melodia."""
    candidates = [
        (degree, Chord((key.tonic_pc + degree) % 12, intervals))
        for degree, intervals in DIATONIC_CHORDS[key.mode]
    ]
    tonic_chord = candidates[0][1]

    progression: List[Chord] = []
    for bar, pc_weights in enumerate(_bar_pitch_class_weights(individual, total_bars)):
        is_last_bar = bar == total_bars - 1
        total = pc_weights.sum()
        if is_last_bar or total == 0:
            # Compasso final resolve na tônica; compasso vazio mantém a harmonia anterior
            progression.append(tonic_chord if is_last_bar or not progression else progression[-1])
            continue

        previous = progression[-1] if progression else None
        best = max(candidates, key=lambda item: _chord_score(*item, pc_weights / total, previous))
        progression.append(best[1])
    return progression


def _nearest_pitch(pitch_class: int, center: int) -> int:
    """Pitch com a pitch class pedida mais próximo de `center` (voicing com movimento mínimo)."""
    base = center - ((center - pitch_class) % 12)
    return base if center - base <= 6 else base + 12


def _humanize(instrument: pretty_midi.Instrument, rng: np.random.Generator, bar_duration: float) -> None:
    """Desloca levemente cada nota no tempo, varia a intensidade e acentua o tempo 1 do compasso."""
    jitter = DRUM_TIMING_JITTER_SECONDS if instrument.is_drum else TIMING_JITTER_SECONDS
    for note in instrument.notes:
        on_downbeat = (note.start % bar_duration) < 1e-6
        shift = rng.uniform(-jitter, jitter)
        duration = note.end - note.start
        note.start = max(0.0, note.start + shift)
        note.end = note.start + max(MIN_NOTE_SECONDS, duration + rng.uniform(-jitter, jitter))

        accent = DOWNBEAT_ACCENT if on_downbeat else 0
        velocity = note.velocity + accent + rng.integers(-VELOCITY_JITTER, VELOCITY_JITTER + 1)
        note.velocity = int(np.clip(velocity, 1, 127))


class MusicArranger:
    """Gera um arranjo harmonizado a partir da melodia do Algoritmo Genético."""

    @staticmethod
    def create_full_band_midi(
        melody_individual: Individual,
        pitch_class_distribution: Sequence[float],
        style: StylePreset,
        groove: Groove,
        bpm: float = 120.0,
        include_drums: bool = True,
        include_bass: bool = True,
        include_harmony: bool = True,
        key: Optional[Key] = None,
        humanize_seed: int = 0,
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        step_duration = (60.0 / bpm) / STEPS_PER_BEAT
        total_steps = sum(g.duration_steps for g in melody_individual.genes)
        total_bars = max(1, -(-total_steps // STEPS_PER_BAR))  # arredonda para cima

        key = key or detect_key(pitch_class_distribution)
        progression = harmonize(melody_individual, key, total_bars)
        kick_hits = groove.hits("kick", GROOVE_HIT_THRESHOLD) or [(0, 1.0)]

        pm.instruments.append(MusicArranger._melody_track(melody_individual, style, step_duration))
        if include_harmony and style.harmony_program is not None:
            pm.instruments.append(MusicArranger._harmony_track(progression, style, step_duration))
        if include_bass:
            pm.instruments.append(MusicArranger._bass_track(progression, style, kick_hits, step_duration))
        if include_drums:
            pm.instruments.append(MusicArranger._drum_track(total_bars, style, groove, step_duration))

        rng = np.random.default_rng(humanize_seed)
        bar_duration = STEPS_PER_BAR * step_duration
        for instrument in pm.instruments:
            _humanize(instrument, rng, bar_duration)
        return pm

    @staticmethod
    def _melody_track(individual: Individual, style: StylePreset, step_duration: float) -> pretty_midi.Instrument:
        lead = pretty_midi.Instrument(program=style.lead_program, name="Lead")
        t = 0.0
        for gene in individual.genes:
            duration = gene.duration_steps * step_duration
            if not gene.is_rest:
                lead.notes.append(pretty_midi.Note(
                    velocity=min(127, gene.velocity + 10), pitch=gene.pitch, start=t, end=t + duration * 0.92
                ))
            t += duration
        return lead

    @staticmethod
    def _harmony_track(progression: List[Chord], style: StylePreset, step_duration: float) -> pretty_midi.Instrument:
        """Acordes sustentados, discretos, abaixo da melodia (região de C3 a C4)."""
        harmony = pretty_midi.Instrument(program=style.harmony_program, name="Harmony")
        bar_duration = STEPS_PER_BAR * step_duration
        voicing_center = 55  # G3
        for bar, chord in enumerate(progression):
            start = bar * bar_duration
            for pc in chord.pitch_classes:
                harmony.notes.append(pretty_midi.Note(
                    velocity=55, pitch=_nearest_pitch(pc, voicing_center), start=start, end=start + bar_duration * 0.95
                ))
        return harmony

    @staticmethod
    def _bass_track(
        progression: List[Chord], style: StylePreset, kick_hits: List[Tuple[int, float]], step_duration: float
    ) -> pretty_midi.Instrument:
        """Baixo travado no bumbo do groove: cada batida de bumbo é uma nota do baixo."""
        bass = pretty_midi.Instrument(program=style.bass_program, name="Bass")
        kick_steps = [step for step, _ in kick_hits]
        next_steps = kick_steps[1:] + [STEPS_PER_BAR + kick_steps[0]]
        for bar, chord in enumerate(progression):
            root = _nearest_pitch(chord.root_pc, center=style.bass_center_pitch)
            fifth = root + chord.intervals[2]
            bar_start_step = bar * STEPS_PER_BAR
            for step, next_step in zip(kick_steps, next_steps):
                on_offbeat = step % STEPS_PER_BEAT != 0
                pitch = fifth if style.bass_fifth_on_offbeats and on_offbeat else root
                start = (bar_start_step + step) * step_duration
                end = start + (next_step - step) * step_duration * style.bass_legato
                bass.notes.append(pretty_midi.Note(velocity=BASS_VELOCITY, pitch=pitch, start=start, end=end))
        return bass

    @staticmethod
    def _drum_track(total_bars: int, style: StylePreset, groove: Groove, step_duration: float) -> pretty_midi.Instrument:
        """Toca o groove compasso a compasso; estilos com `phrase_fills` ganham prato e virada por frase."""
        drums = pretty_midi.Instrument(program=style.drum_kit, is_drum=True, name="Drums")
        part_pitches = {"kick": KICK, "snare": style.snare_pitch, "hat": CLOSED_HAT}

        def hit(pitch: int, step: int, velocity: int) -> None:
            start = step * step_duration
            drums.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start + 0.1))

        def strength_to_velocity(strength: float) -> int:
            return int(DRUM_VELOCITY_MIN + strength * (DRUM_VELOCITY_MAX - DRUM_VELOCITY_MIN))

        for bar in range(total_bars):
            bar_step = bar * STEPS_PER_BAR
            if bar == total_bars - 1:
                # Final: acento único em vez de seguir o loop até o fim
                hit(CRASH if style.phrase_fills else style.snare_pitch, bar_step, 110)
                hit(KICK, bar_step, 115)
                continue

            phrase_position = bar % BARS_PER_PHRASE
            is_fill_bar = style.phrase_fills and phrase_position == BARS_PER_PHRASE - 1
            groove_end = STEPS_PER_BAR // 2 if is_fill_bar else STEPS_PER_BAR
            if style.phrase_fills and phrase_position == 0:
                hit(CRASH, bar_step, 100)

            for part, pitch in part_pitches.items():
                for step, strength in groove.hits(part, GROOVE_HIT_THRESHOLD):
                    if step < groove_end:
                        hit(pitch, bar_step + step, strength_to_velocity(strength))

            if is_fill_bar:
                for i, pitch in enumerate(FILL_PITCHES):
                    hit(pitch, bar_step + groove_end + i, 85 + i * 4)

        return drums
