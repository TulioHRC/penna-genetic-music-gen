"""
Demo Reference MIDI Generator:
Gera arquivos MIDI de exemplo com motivos e riffs característicos de rock / Pitty
(Ex: 'Máscara', 'Equalize', 'Admirável Chip Novo') caso o usuário ainda não tenha
colocado arquivos .mid reais na pasta data/reference_midi.
"""

import pretty_midi


def create_pitty_riff_demo(output_path: str, title: str = "Pitty Riff"):
    pm = pretty_midi.PrettyMIDI(initial_tempo=130.0)
    guitar = pretty_midi.Instrument(program=29, name="Distortion Guitar")  # Overdriven Guitar

    # Riff em Lá Menor / Pentatônica (A, C, D, E, G) - Característico de 'Máscara' / 'Admirável Chip Novo'
    # Pitches: A2=45, C3=48, D3=50, E3=52, G3=55, A3=57, C4=60, D4=62, E4=64
    riff_sequence = [
        (45, 0.0, 0.25),   # A2
        (45, 0.25, 0.50),  # A2
        (48, 0.50, 0.75),  # C3
        (50, 0.75, 1.00),  # D3
        (52, 1.00, 1.50),  # E3
        (50, 1.50, 1.75),  # D3
        (48, 1.75, 2.00),  # C3
        (45, 2.00, 2.50),  # A2
        (55, 2.50, 2.75),  # G3
        (57, 2.75, 3.50),  # A3
        (55, 3.50, 3.75),  # G3
        (52, 3.75, 4.00),  # E3
    ]

    # Repetir riff por 2 compassos
    for note_pitch, start, end in riff_sequence:
        guitar.notes.append(pretty_midi.Note(velocity=105, pitch=note_pitch, start=start, end=end))
        # Segunda repetição com pequena variação
        guitar.notes.append(pretty_midi.Note(velocity=105, pitch=note_pitch, start=start + 4.0, end=end + 4.0))

    pm.instruments.append(guitar)
    pm.write(output_path)
    print(f"[Corpus Demo] Criado MIDI de referência: {output_path}")


if __name__ == "__main__":
    create_pitty_riff_demo("data/reference_midi/pitty_riff_1.mid", "Riff 1 - Mascara Style")
    create_pitty_riff_demo("data/reference_midi/pitty_riff_2.mid", "Riff 2 - Chip Novo Style")
