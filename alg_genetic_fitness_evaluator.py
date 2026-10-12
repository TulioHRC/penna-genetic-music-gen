from typing import Dict, List, Optional
import numpy as np
from corpus_analyzer import StyleProfile
from alg_genetic_model import Individual, Gene


# Limite máximo recomendado de repetições consecutivas da mesma nota antes de aplicar penalidade
MAX_CONSECUTIVE_REPEATED_NOTES = 3

# Salto intervalar máximo aceitável em semitons (17 semitons = oitava + quarta aumentada/quinta diminuta)
MAX_INTERVAL_SEMITONES = 17

# Faixa ideal de proporção de pausas (rests) para garantir respiro melódico sem silêncio excessivo
IDEAL_REST_RATIO_MIN = 0.10
IDEAL_REST_RATIO_MAX = 0.35
IDEAL_REST_TARGET = 0.22
REST_PENALTY_MULTIPLIER = 2.5

# Número de semitons em 2 oitavas (usado para normalizar desvio de tessitura média)
OCTAVE_SEMITONES_SPAN = 24.0

# Slots temporais por compasso (em compasso 4/4 com subdivisão em semicolcheias: 4 batidas * 4 = 16 steps)
STEPS_PER_BAR = 16


class CorpusFitnessEvaluator:
    """
    Função de fitness multiobjetivo guiada por estatísticas do corpus e princípios composicionais:
    1. Pitch-Class Fit (w_pitch_class): fidelidade às notas e escalas do estilo (ex.: pentatônica/menor da Pitty).
    2. Interval Transition Fit (w_interval): fidelidade aos saltos melódicos característicos do estilo.
    3. Riff/Motif Recurrence (w_motif): repetição e coerência de motivos melódicos entre compassos.
    4. Tonal Resolution (w_resolution): cadência melódica resolvendo em notas fundamentais nos finais de frase.
    5. Register Fit (w_register): respeito à tessitura do instrumento/vocalista.
    6. Rhythm & Phrasing (w_rhythm): equilíbrio dinâmico entre notas ativas e pausas.
    """

    def __init__(
        self,
        profile: StyleProfile,
        w_pitch_class: float = 0.25,
        w_interval: float = 0.25,
        w_motif: float = 0.20,
        w_resolution: float = 0.10,
        w_register: float = 0.10,
        w_rhythm: float = 0.10,
        tonic_pitch_class: Optional[int] = None
    ):
        self.profile = profile

        # Normalização dos pesos para garantir que somem exatamente 1.0
        total_weight = w_pitch_class + w_interval + w_motif + w_resolution + w_register + w_rhythm
        if total_weight <= 0:
            total_weight = 1.0

        self.w_pitch_class = w_pitch_class / total_weight
        self.w_interval = w_interval / total_weight
        self.w_motif = w_motif / total_weight
        self.w_resolution = w_resolution / total_weight
        self.w_register = w_register / total_weight
        self.w_rhythm = w_rhythm / total_weight

        # Tônica do tom (se não fornecida explicitamente, usa a pitch class mais frequente do corpus)
        if tonic_pitch_class is not None:
            self.tonic_pc = tonic_pitch_class % 12
        else:
            self.tonic_pc = int(np.argmax(self.profile.pitch_class_distribution))

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "pitch": self.w_pitch_class,
            "interval": self.w_interval,
            "motif": self.w_motif,
            "resolution": self.w_resolution,
            "register": self.w_register,
            "rhythm": self.w_rhythm,
        }

    def evaluate(self, individual: Individual) -> float:
        scores = self.score_breakdown(individual)
        if scores is None:
            return 0.001  # Sem notas ativas
        total_fitness = sum(self.weights[name] * score for name, score in scores.items())
        return float(np.clip(total_fitness, 0.0001, 1.0))

    def score_breakdown(self, individual: Individual) -> Optional[Dict[str, float]]:
        active_notes = [g for g in individual.genes if not g.is_rest]

        if not active_notes:
            return None

        # 1. Pitch-Class Fitness (Frequência das notas em relação ao corpus)
        pc_scores = [self.profile.pitch_class_distribution[g.pitch % 12] for g in active_notes]
        pitch_score = float(np.mean(pc_scores)) if pc_scores else 0.0
        max_pc_prob = float(np.max(self.profile.pitch_class_distribution)) + 1e-6
        pitch_fit = min(1.0, pitch_score / max_pc_prob)

        # 2. Interval Transition Fitness (Fluência melódica e saltos)
        interval_scores = []
        consecutive_same_note = 0
        monotonous_transitions = 0
        excessive_jumps = 0

        for i in range(1, len(active_notes)):
            interval = active_notes[i].pitch - active_notes[i - 1].pitch
            prob = self.profile.interval_transition_distribution.get(interval, 0.001)
            interval_scores.append(prob)

            if interval == 0:
                consecutive_same_note += 1
                if consecutive_same_note > MAX_CONSECUTIVE_REPEATED_NOTES:
                    monotonous_transitions += 1
            else:
                consecutive_same_note = 0

            # Penaliza saltos acima da extensão aceitável
            if abs(interval) > MAX_INTERVAL_SEMITONES:
                excessive_jumps += 1

        # Relativo ao intervalo mais comum da referência (1.0 = só usa os intervalos mais típicos)
        max_interval_prob = max(self.profile.interval_transition_distribution.values(), default=1.0)
        interval_similarity = float(np.mean(interval_scores)) / max_interval_prob if interval_scores else 0.1
        # Penalidades multiplicativas e proporcionais: cada salto/repetição a menos sempre melhora
        transitions = max(1, len(active_notes) - 1)
        jump_rate = excessive_jumps / transitions
        monotony_rate = monotonous_transitions / transitions
        interval_fit = float(np.clip(interval_similarity * (1 - jump_rate) * (1 - monotony_rate), 0.0, 1.0))

        # 3. Motif / Riff Recurrence Fitness (Coerência e repetição de ideias)
        motif_fit = self._evaluate_motif_recurrence(individual.genes)

        # 4. Tonal Resolution / Cadence Fitness (Finalização na tônica/quinta)
        last_note_pc = active_notes[-1].pitch % 12
        dominant_pc = (self.tonic_pc + 7) % 12  # Quinta justa
        mediant_pc = (self.tonic_pc + 3) % 12   # Terça menor

        if last_note_pc == self.tonic_pc:
            resolution_fit = 1.0
        elif last_note_pc in (dominant_pc, mediant_pc):
            resolution_fit = 0.8
        elif self.profile.pitch_class_distribution[last_note_pc] > 0.08:
            resolution_fit = 0.5
        else:
            resolution_fit = 0.1

        # 5. Register Fit (Tessitura do estilo)
        pitches = [g.pitch for g in active_notes]
        mean_p = float(np.mean(pitches))
        reg_diff = abs(mean_p - self.profile.avg_pitch) / OCTAVE_SEMITONES_SPAN
        out_of_bounds = sum(1 for p in pitches if p < self.profile.min_pitch or p > self.profile.max_pitch)
        out_penalty = out_of_bounds / len(pitches)
        register_fit = max(0.0, 1.0 - reg_diff - out_penalty)

        # 6. Rhythm & Phrasing Balance (Proporção de notas e pausas)
        rest_ratio = sum(1 for g in individual.genes if g.is_rest) / len(individual.genes)
        if IDEAL_REST_RATIO_MIN <= rest_ratio <= IDEAL_REST_RATIO_MAX:
            rhythm_fit = 1.0
        else:
            rhythm_fit = max(0.0, 1.0 - abs(rest_ratio - IDEAL_REST_TARGET) * REST_PENALTY_MULTIPLIER)

        return {
            "pitch": pitch_fit,
            "interval": interval_fit,
            "motif": motif_fit,
            "resolution": resolution_fit,
            "register": register_fit,
            "rhythm": rhythm_fit,
        }

    def _evaluate_motif_recurrence(self, genes: List[Gene]) -> float:
        pitches = [g.pitch if not g.is_rest else -1 for g in genes]
        if len(pitches) < 8:
            return 0.5

        # Cria n-gramas de tamanho 4 (motivos de 4 eventos)
        motif_size = 4
        motifs = [tuple(pitches[i:i + motif_size]) for i in range(len(pitches) - motif_size + 1)]
        if not motifs:
            return 0.5

        # Conta motivos repetidos (sem considerar repetição trivial só de pausas)
        total_motifs = len(motifs)
        unique_motifs = len(set(motifs))
        repetition_rate = 1.0 - (unique_motifs / total_motifs)

        if 0.20 <= repetition_rate <= 0.55:
            return 1.0
        elif repetition_rate < 0.20:
            return float(np.clip(repetition_rate / 0.20, 0.2, 1.0))
        else:
            return float(np.clip(1.0 - (repetition_rate - 0.55) * 2.0, 0.2, 1.0))
