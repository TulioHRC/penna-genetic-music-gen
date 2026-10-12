"""
SoundFont Renderer:
Renderiza um PrettyMIDI usando samples reais de instrumentos (arquivo .sf2) via tinysoundfont,
que traz o próprio motor de síntese (não precisa de FluidSynth instalado no sistema).

Bem mais natural que a síntese aditiva de audio_synth.py: cada nota usa a gravação de um
instrumento de verdade, com envelope, vibrato e variação de timbre por velocity.
"""

from typing import List, Tuple

import numpy as np
import pretty_midi
import scipy.signal as signal

DRUM_CHANNEL = 9
MELODIC_CHANNELS = [c for c in range(16) if c != DRUM_CHANNEL]
TAIL_SECONDS = 1.5  # deixa as notas finais e o reverb soarem até o fim

# Reverb de sala por convolução com resposta ao impulso sintética (ruído com decaimento exponencial)
REVERB_SECONDS = 1.2
REVERB_DECAY_PER_SECOND = 5.0
REVERB_WET = 0.18
PEAK_LEVEL = 0.9

NOTE_OFF, NOTE_ON = 0, 1  # note-off antes de note-on no mesmo instante (nota repetida)


def soundfont_available() -> bool:
    try:
        import tinysoundfont  # noqa: F401
        return True
    except ImportError:
        return False


def _collect_events(pm: pretty_midi.PrettyMIDI) -> Tuple[List[tuple], List[tuple]]:
    """Atribui um canal MIDI a cada instrumento e devolve (programas por canal, eventos ordenados)."""
    channel_programs, events = [], []
    melodic_channels = iter(MELODIC_CHANNELS)
    for instrument in pm.instruments:
        channel = DRUM_CHANNEL if instrument.is_drum else next(melodic_channels)
        channel_programs.append((channel, instrument.program, instrument.is_drum))
        for note in instrument.notes:
            events.append((note.start, NOTE_ON, channel, note.pitch, note.velocity))
            events.append((note.end, NOTE_OFF, channel, note.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))
    return channel_programs, events


def _room_reverb(stereo: np.ndarray, sr: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(REVERB_SECONDS * sr)) / sr
    impulse = rng.standard_normal((len(t), 2)) * np.exp(-REVERB_DECAY_PER_SECOND * t)[:, None]
    b, a = signal.butter(1, 5000, btype="lowpass", fs=sr)  # reverb mais escuro, como uma sala real
    impulse = signal.lfilter(b, a, impulse, axis=0)
    impulse /= np.sqrt(np.sum(impulse ** 2, axis=0))
    wet = signal.fftconvolve(stereo, impulse, axes=0)[: len(stereo)]
    return (1 - REVERB_WET) * stereo + REVERB_WET * wet


def render_soundfont_dry(pm: pretty_midi.PrettyMIDI, soundfont_path: str, sr: int = 44100) -> np.ndarray:
    """Áudio estéreo (n_amostras, 2) sem reverb nem normalização, para ser mixado com outras fontes."""
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=sr)
    soundfont_id = synth.sfload(soundfont_path)

    channel_programs, events = _collect_events(pm)
    for channel, program, is_drum in channel_programs:
        synth.program_select(channel, soundfont_id, 0, program, is_drums=is_drum)  # bateria: program = kit

    chunks = []
    cursor = 0
    for time, kind, channel, pitch, velocity in events:
        target = int(time * sr)
        if target > cursor:
            chunks.append(np.frombuffer(synth.generate(target - cursor), dtype=np.float32).copy())
            cursor = target
        if kind == NOTE_ON:
            synth.noteon(channel, pitch, velocity)
        else:
            synth.noteoff(channel, pitch)
    chunks.append(np.frombuffer(synth.generate(int(TAIL_SECONDS * sr)), dtype=np.float32).copy())
    return np.concatenate(chunks).reshape(-1, 2)


def master(stereo: np.ndarray, sr: int) -> np.ndarray:
    """Reverb de sala + normalização de pico."""
    stereo = _room_reverb(stereo, sr)
    peak = np.max(np.abs(stereo)) + 1e-9
    return (stereo / peak * PEAK_LEVEL).astype(np.float32)


def render_with_soundfont(pm: pretty_midi.PrettyMIDI, soundfont_path: str, sr: int = 44100) -> np.ndarray:
    """Retorna áudio estéreo float32 (n_amostras, 2) normalizado."""
    return master(render_soundfont_dry(pm, soundfont_path, sr), sr)
