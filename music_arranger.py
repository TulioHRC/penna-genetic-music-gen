"""
Arranger and Multi-Track MIDI Generator:
Transforma a melodia evoluída pelo Algoritmo Genético em um arranjo completo de banda:
1. Linha Melódica Principal (Guitarra Solo / Overdriven Guitar)
2. Trompete / Metais (Brass / Trumpet acompanhando frases e acentos)
3. Baixo Elétrico (Electric Bass marcando as tônicas)
4. Bateria de Rock (Drums Track no canal MIDI 10: bumbo, caixa, chimbal)
"""

from typing import List
import pretty_midi
from genetic_model import Gene, Individual


class MusicArranger:
    """Gera um arranjo de banda completa a partir da melodia do Algoritmo Genético."""

    @staticmethod
    def create_full_band_midi(
        melody_individual: Individual,
        bpm: float = 128.0,
        include_drums: bool = True,
        include_bass: bool = True,
        include_brass: bool = True
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        step_duration = (60.0 / bpm) / 4.0  # step de semicolcheia (16 por compasso)
        total_duration = sum(g.duration_steps for g in melody_individual.genes) * step_duration

        # 1. Pista de Guitarra (Overdriven Guitar - GM Program 29)
        guitar = pretty_midi.Instrument(program=29, name="Lead Guitar")
        curr_t = 0.0
        for g in melody_individual.genes:
            dur = g.duration_steps * step_duration
            if not g.is_rest:
                guitar.notes.append(pretty_midi.Note(
                    velocity=min(127, g.velocity + 10),
                    pitch=g.pitch,
                    start=curr_t,
                    end=curr_t + (dur * 0.92)
                ))
            curr_t += dur
        pm.instruments.append(guitar)

        # 2. Pista de Trompete / Metais (Trumpet - GM Program 56 ou Brass Section 61)
        if include_brass:
            trumpet = pretty_midi.Instrument(program=56, name="Trumpet / Brass")
            curr_t = 0.0
            for g in melody_individual.genes:
                dur = g.duration_steps * step_duration
                # O trompete toca nas notas mais longas (duração >= 4 steps = semínima) para criar ataques dramáticos
                if not g.is_rest and g.duration_steps >= 3:
                    trumpet.notes.append(pretty_midi.Note(
                        velocity=min(115, g.velocity + 5),
                        pitch=g.pitch + 12 if g.pitch < 65 else g.pitch,  # registro brilhante de metais
                        start=curr_t,
                        end=curr_t + (dur * 0.85)
                    ))
                curr_t += dur
            pm.instruments.append(trumpet)

        # 3. Pista de Baixo Elétrico (Electric Bass Finger - GM Program 33)
        if include_bass:
            bass = pretty_midi.Instrument(program=33, name="Electric Bass")
            # Agrupa por compasso (a cada 16 steps = 4 batidas) ou notas principais
            curr_t = 0.0
            for g in melody_individual.genes:
                dur = g.duration_steps * step_duration
                if not g.is_rest and g.duration_steps >= 2:
                    # Toca na oitava grave (oitavas 1 e 2, pitch entre 33 e 50)
                    bass_pitch = (g.pitch % 12) + 36  # C2 a B2
                    bass.notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=bass_pitch,
                        start=curr_t,
                        end=curr_t + (dur * 0.90)
                    ))
                curr_t += dur
            pm.instruments.append(bass)

        # 4. Pista de Bateria (Standard Rock Kit no canal 10: is_drum=True)
        if include_drums:
            drums = pretty_midi.Instrument(program=0, is_drum=True, name="Rock Drum Kit")
            # GM Drum Map:
            # 36: Bass Drum 1 (Bumbo)
            # 38: Acoustic Snare (Caixa)
            # 42: Closed Hi-Hat (Chimbal fechado)
            # 46: Open Hi-Hat
            # 49: Crash Cymbal (Prato de ataque)

            beat_dur = 60.0 / bpm
            current_bar_t = 0.0

            # Prato de ataque no primeiro tempo
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=49, start=0.0, end=0.5))

            while current_bar_t < total_duration:
                # Loop de bateria rock padrão por compasso (4 batidas)
                for beat in range(4):
                    t = current_bar_t + (beat * beat_dur)
                    if t >= total_duration:
                        break

                    # Hi-hat em todas as colcheias (tempo e contratempo)
                    drums.notes.append(pretty_midi.Note(velocity=80, pitch=42, start=t, end=t + 0.1))
                    drums.notes.append(pretty_midi.Note(velocity=70, pitch=42, start=t + (beat_dur / 2), end=t + (beat_dur / 2) + 0.1))

                    # Bumbo (Kick) nos tempos 1 e 3
                    if beat == 0 or beat == 2:
                        drums.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=t, end=t + 0.15))

                    # Caixa (Snare) nos tempos 2 e 4 (backbeat clássico do rock)
                    if beat == 1 or beat == 3:
                        drums.notes.append(pretty_midi.Note(velocity=115, pitch=38, start=t, end=t + 0.15))

                current_bar_t += (4 * beat_dur)

            pm.instruments.append(drums)

        return pm
