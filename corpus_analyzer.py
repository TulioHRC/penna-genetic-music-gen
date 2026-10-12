"""
Reference Corpus Analyzer:
Extrai perfil estatístico e padrões estilísticos de músicas de referência (arquivos MIDI).
Analisa:
- Distribuição de Pitch Classes (histograma das 12 classes de notas).
- Transições de Intervalos Melódicos (bi-gramas de semitons entre notas consecutivas).
- Distribuição de Durações Rítmicas (em frações de batida / 16th notes).
- Faixa típica de pitch (min, max, média).
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import pretty_midi


@dataclass
class StyleProfile:
    pitch_class_distribution: np.ndarray = field(default_factory=lambda: np.ones(12) / 12.0)
    interval_transition_distribution: Dict[int, float] = field(default_factory=dict)
    duration_distribution: Dict[float, float] = field(default_factory=dict)
    avg_pitch: float = 60.0
    min_pitch: int = 40
    max_pitch: int = 84
    density_notes_per_bar: float = 8.0
    tempo_bpm: float = 120.0


class CorpusAnalyzer:
    """Extrai características de estilo de um ou mais arquivos MIDI."""

    def __init__(self, midi_paths: List[str]):
        self.midi_paths = midi_paths

    def build_profile(self) -> StyleProfile:
        all_pitches: List[int] = []
        all_durations: List[float] = []
        intervals: List[int] = []
        tempos: List[float] = []

        for path in self.midi_paths:
            try:
                pm = pretty_midi.PrettyMIDI(path)
                tempos.append(float(pm.get_tempo_changes()[1][0]))
                for instrument in pm.instruments:
                    if instrument.is_drum:
                        continue  # Analisamos instrumentos melódicos/harmônicos
                    
                    notes = sorted(instrument.notes, key=lambda n: n.start)
                    for i, note in enumerate(notes):
                        all_pitches.append(note.pitch)
                        duration = round(note.end - note.start, 2)
                        all_durations.append(duration)
                        if i > 0:
                            prev_note = notes[i - 1]
                            # Apenas calcula intervalo melódico se notas estiverem razoavelmente próximas
                            if note.start - prev_note.end <= 1.0:
                                interval = note.pitch - prev_note.pitch
                                intervals.append(interval)
            except Exception as e:
                print(f"[Aviso] Erro ao carregar {path}: {e}")

        # Se nenhum MIDI foi carregado com sucesso, cria um perfil padrão de Rock / Pentatônica menor
        if not all_pitches:
            return self._default_rock_profile()

        # 1. Distribuição de pitch class (0-11)
        pc_counts = np.zeros(12)
        for p in all_pitches:
            pc_counts[p % 12] += 1
        pc_dist = (pc_counts + 1e-4) / (np.sum(pc_counts) + 12 * 1e-4)

        # 2. Distribuição de transições de intervalo
        interval_counter = Counter(intervals)
        total_intervals = sum(interval_counter.values()) if intervals else 1
        interval_dist = {k: v / total_intervals for k, v in interval_counter.items()}

        # 3. Distribuição de durações aproximadas
        duration_counter = Counter([round(d, 2) for d in all_durations])
        total_dur = sum(duration_counter.values()) if all_durations else 1
        dur_dist = {k: v / total_dur for k, v in duration_counter.items()}

        return StyleProfile(
            pitch_class_distribution=pc_dist,
            interval_transition_distribution=interval_dist,
            duration_distribution=dur_dist,
            avg_pitch=float(np.mean(all_pitches)),
            min_pitch=int(np.min(all_pitches)),
            max_pitch=int(np.max(all_pitches)),
            density_notes_per_bar=len(all_pitches) / max(1, len(self.midi_paths) * 8),
            tempo_bpm=float(np.median(tempos))
        )

    def _default_rock_profile(self) -> StyleProfile:
        """Perfil fallback (Rock em Lá Menor / Pentatônica A Minor)."""
        # A, C, D, E, G
        # Pitch classes: A=9, B=11, C=0, D=2, E=4, F=5, G=7
        pc = np.zeros(12)
        rock_scale = [9, 0, 2, 4, 7]  # A minor pentatonic
        for p in rock_scale:
            pc[p] = 1.0
        pc = pc / np.sum(pc)

        # Intervalos comuns de rock: uníssono (0), segunda (2, -2), terça menor (3, -3), quarta justa (5, -5)
        int_dist = {0: 0.15, 2: 0.2, -2: 0.2, 3: 0.15, -3: 0.15, 5: 0.08, -5: 0.07}
        return StyleProfile(
            pitch_class_distribution=pc,
            interval_transition_distribution=int_dist,
            duration_distribution={0.25: 0.3, 0.5: 0.4, 1.0: 0.2, 2.0: 0.1},
            avg_pitch=64.0,
            min_pitch=45,
            max_pitch=79
        )
