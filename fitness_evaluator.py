"""
Corpus-Guided Fitness Evaluator:
Calcula a adaptação (fitness) do indivíduo comparando sua estrutura musical
às distribuições extraídas do corpus de referência (Pitty/Rock).
"""

from typing import Tuple
import numpy as np
from corpus_analyzer import StyleProfile
from genetic_model import Individual


class CorpusFitnessEvaluator:
    """
    Função de fitness multiobjetivo guiada por estatísticas do corpus.
    Combina:
    1. Pitch-Class Fit: aderência às notas e escalas mais frequentes na referência.
    2. Interval Transition Fit: proximidade da distribuição de saltos melódicos.
    3. Register Fit: respeito à tessitura (faixa de notas min/max/média).
    4. Rest & Phrasing Fit: penalização de ausência total de pausas ou pausas em excesso.
    5. Melodic Contour Penalty: penalização de repetições excessivas da mesma nota.
    """

    def __init__(
        self,
        profile: StyleProfile,
        w_pitch_class: float = 0.35,
        w_interval: float = 0.35,
        w_register: float = 0.15,
        w_rhythm: float = 0.15
    ):
        self.profile = profile
        self.w_pitch_class = w_pitch_class
        self.w_interval = w_interval
        self.w_register = w_register
        self.w_rhythm = w_rhythm

    def evaluate(self, individual: Individual) -> float:
        active_notes = [g for g in individual.genes if not g.is_rest]

        if not active_notes:
            return 0.001  # Sem notas ativas

        # 1. Pitch-Class Fitness
        # Verifica quão frequentes as classes das notas do indivíduo são no perfil de referência
        pc_scores = []
        for g in active_notes:
            pc = g.pitch % 12
            pc_scores.append(self.profile.pitch_class_distribution[pc])
        pitch_score = float(np.mean(pc_scores)) if pc_scores else 0.0
        # Normalizar para escala aproximada de [0, 1]
        max_pc_prob = np.max(self.profile.pitch_class_distribution) + 1e-6
        pitch_fit = min(1.0, pitch_score / max_pc_prob)

        # 2. Interval Transition Fitness
        # Compara cada salto melódico sucessivo com o padrão das músicas de referência
        interval_scores = []
        consecutive_same_note = 0
        monotony_penalty = 0.0

        for i in range(1, len(active_notes)):
            interval = active_notes[i].pitch - active_notes[i - 1].pitch
            prob = self.profile.interval_transition_distribution.get(interval, 0.001)
            interval_scores.append(prob)

            if interval == 0:
                consecutive_same_note += 1
                if consecutive_same_note > 3:
                    monotony_penalty += 0.1
            else:
                consecutive_same_note = 0

            # Penalidade para saltos dissonantes ou absurdos (> 1 oitava e meia = 18 semitons)
            if abs(interval) > 17:
                monotony_penalty += 0.15

        interval_fit = float(np.mean(interval_scores)) * 5.0 if interval_scores else 0.5
        interval_fit = max(0.0, min(1.0, interval_fit - monotony_penalty))

        # 3. Register Fit (Tessitura)
        pitches = [g.pitch for g in active_notes]
        mean_p = np.mean(pitches)
        reg_diff = abs(mean_p - self.profile.avg_pitch) / 24.0  # desvio relativo a 2 oitavas
        out_of_bounds = sum(1 for p in pitches if p < self.profile.min_pitch or p > self.profile.max_pitch)
        out_penalty = out_of_bounds / len(pitches)
        register_fit = max(0.0, 1.0 - reg_diff - out_penalty)

        # 4. Rhythm & Rest Balance Fit
        # Proporção saudável de notas vs pausas (músicas de rock precisam de respiro e fraseado)
        rest_ratio = sum(1 for g in individual.genes if g.is_rest) / len(individual.genes)
        # Faixa ideal de pausas entre 10% e 35%
        if 0.10 <= rest_ratio <= 0.35:
            rhythm_fit = 1.0
        else:
            rhythm_fit = max(0.0, 1.0 - abs(rest_ratio - 0.22) * 2.5)

        total_fitness = (
            self.w_pitch_class * pitch_fit
            + self.w_interval * interval_fit
            + self.w_register * register_fit
            + self.w_rhythm * rhythm_fit
        )

        return float(np.clip(total_fitness, 0.0001, 1.0))
