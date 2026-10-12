import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Gene:
    # Representa um evento musical em um slot temporal (step).

    pitch: int # nota MIDI (0 a 127)
    duration_steps: int = 2  # duração medida em steps (ex: 1 step = 1 semicolcheia)
    velocity: int = 90 # intensidade da nota (0 a 127)
    is_rest: bool = False # se True, o gene é uma pausa (silêncio)


@dataclass
class Individual:
    genes: List[Gene] = field(default_factory=list)
    fitness: float = 0.0

    @classmethod
    def criar_random(
        individual_class,
        total_steps: int = 64, # 4 compassos de 4/4 com 16 steps por compasso
        allowed_pitches: List[int] = None,
        rest_prob: float = 0.2
    ) -> "Individual":
        if allowed_pitches is None:
            # Fallback escala cromática intermediária (D3 a A5)
            allowed_pitches = list(range(50, 77))

        genes = []
        current_step = 0
        while current_step < total_steps:
            # Duração em steps: 1 (semicolcheia), 2 (colcheia), 4 (semínima), 8 (mínima) -- ponderada para mais colcheias e semínimas
            dur = random.choice([1, 2, 2, 4, 4, 8])
            if current_step + dur > total_steps:
                dur = total_steps - current_step

            is_rest = random.random() < rest_prob
            pitch = random.choice(allowed_pitches)
            velocity = random.randint(70, 110) # aleatório para dar uma variação adicional

            genes.append(Gene(
                pitch=pitch,
                duration_steps=dur,
                velocity=velocity,
                is_rest=is_rest
            ))
            current_step += dur

        return individual_class(genes=genes)
