"""
Genetic Algorithm Data Model:
Define o genótipo, cromossomo, indivíduo e conversor de/para MIDI.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple
import pretty_midi


@dataclass
class Gene:
    """
    Representa um evento musical em um slot temporal (step).
    - pitch: nota MIDI (0 a 127).
    - duration_steps: duração medida em steps (ex: 1 step = 1 semicolcheia).
    - velocity: intensidade (dinâmica, 0-127).
    - is_rest: se True, o gene é uma pausa (silêncio).
    """
    pitch: int
    duration_steps: int = 2  # padrão colcheia em grid de 16 steps por compasso
    velocity: int = 90
    is_rest: bool = False


@dataclass
class Individual:
    """
    Indivíduo do Algoritmo Genético.
    Possui uma sequência de genes (linha melódica ao longo dos compassos).
    """
    genes: List[Gene] = field(default_factory=list)
    fitness: float = 0.0

    @classmethod
    def create_random(
        cls,
        total_steps: int = 64,  # 4 compassos de 16 steps
        allowed_pitches: List[int] = None,
        rest_prob: float = 0.2
    ) -> "Individual":
        if allowed_pitches is None:
            # Fallback escala cromática intermediária (D3 a A5)
            allowed_pitches = list(range(50, 77))

        genes = []
        current_step = 0
        while current_step < total_steps:
            # Duração em steps: 1 (semicolcheia), 2 (colcheia), 4 (semínima), 8 (mínima)
            dur = random.choice([1, 2, 2, 4, 4, 8])
            if current_step + dur > total_steps:
                dur = total_steps - current_step

            is_rest = random.random() < rest_prob
            pitch = random.choice(allowed_pitches)
            velocity = random.randint(70, 110)

            genes.append(Gene(
                pitch=pitch,
                duration_steps=dur,
                velocity=velocity,
                is_rest=is_rest
            ))
            current_step += dur

        return cls(genes=genes)

    def to_pretty_midi(
        self,
        bpm: float = 120.0,
        program: int = 27  # 27 = Electric Guitar (clean) ou 29 = Overdriven Guitar
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(program=program, name="Melody")

        # Em compasso 4/4 a 120 bpm, cada batida (semínima) dura 0.5s.
        # Dividindo a semínima em 4 steps (grid de 16 steps por compasso):
        step_duration = (60.0 / bpm) / 4.0

        current_time = 0.0
        for gene in self.genes:
            duration_seconds = gene.duration_steps * step_duration
            if not gene.is_rest:
                note = pretty_midi.Note(
                    velocity=gene.velocity,
                    pitch=gene.pitch,
                    start=current_time,
                    end=current_time + (duration_seconds * 0.95)  # ligeiro release para articulação
                )
                instrument.notes.append(note)
            current_time += duration_seconds

        pm.instruments.append(instrument)
        return pm

    def save_midi(self, filepath: str, bpm: float = 120.0, program: int = 27) -> None:
        pm = self.to_pretty_midi(bpm=bpm, program=program)
        pm.write(filepath)
