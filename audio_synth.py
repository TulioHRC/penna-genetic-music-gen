"""
Audio Synthesizer & Analog Master Processor:
- Sintetiza arquivos MIDI em áudio.
- Aplica processamento acústico/DSP para dar calor analógico e presença de instrumento real:
  • Saturação analógica suave (harmônicos ímpares de válvula)
  • EQ e modelagem de presença de amplificador de guitarra/banda
  • Reverb / ambiência de sala de estúdio
- Exporta em WAV e MP3 de alta definição.
"""

import os
import glob
from typing import Optional
import numpy as np
import pretty_midi
import soundfile as sf
import scipy.signal as signal
from scipy.io import wavfile


def apply_analog_warmth_and_space(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Aplica DSP para transformar a síntese crua em um som quente e encorpado:
    - Saturação suave (analog drive).
    - Modelagem de curva de equalização (presença nos médios e agudos arredondados).
    - Early reflections / reverberação de sala.
    """
    if len(audio) == 0:
        return audio

    # 1. Normalizar entrada
    peak = np.max(np.abs(audio)) + 1e-6
    x = audio / peak

    # 2. Saturação harmônica não-linear (simulação valvulada: y = tanh(k*x))
    drive = 1.6
    saturated = np.tanh(drive * x)

    # 3. Modelagem de EQ de estúdio (Filtro passa-faixa suave eliminando ruído subgrave e estridência digital)
    # 50 Hz a 7500 Hz
    sos = signal.butter(2, [50, 7500], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, saturated)

    # 4. Ambiência de Estúdio (comb delays para dar profundidade e tirar sensação estéril)
    delays_ms = [23, 31, 47]
    gains = [0.22, 0.16, 0.11]
    reverb_tail = np.zeros_like(filtered)
    for d_ms, g in zip(delays_ms, gains):
        d_samples = int(sr * d_ms / 1000.0)
        if len(filtered) > d_samples:
            reverb_tail[d_samples:] += g * filtered[:-d_samples]

    mastered = 0.82 * filtered + 0.28 * reverb_tail

    # 5. Normalização final com headroom seguro (-0.5 dB)
    mastered_peak = np.max(np.abs(mastered)) + 1e-6
    final_audio = (mastered / mastered_peak) * 0.94

    return final_audio


def convert_midi_to_audio(
    midi_path: str,
    output_dir: Optional[str] = None,
    output_format: str = "mp3",
    samplerate: int = 44100,
    apply_dsp: bool = True
) -> str:
    """Converte MIDI para WAV e MP3 com processamento acústico."""
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"Arquivo MIDI não encontrado: {midi_path}")

    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    target_dir = output_dir if output_dir else os.path.dirname(midi_path)
    os.makedirs(target_dir, exist_ok=True)

    wav_out = os.path.join(target_dir, f"{base_name}.wav")
    mp3_out = os.path.join(target_dir, f"{base_name}.mp3")

    pm = pretty_midi.PrettyMIDI(midi_path)
    audio = pm.synthesize(fs=samplerate)

    if apply_dsp:
        audio = apply_analog_warmth_and_space(audio, sr=samplerate)
    else:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val + 1e-4)

    # Salva WAV (16-bit PCM)
    wavfile.write(wav_out, samplerate, (audio * 32767).astype(np.int16))

    # Codifica MP3
    data, sr = sf.read(wav_out)
    sf.write(mp3_out, data, sr, format="MP3")

    print(f"  🔊 Áudio gerado: {mp3_out} (WAV: {wav_out})")
    return mp3_out if output_format.lower() == "mp3" else wav_out


def convert_all_in_folder(folder: str):
    midi_files = glob.glob(os.path.join(folder, "*.mid"))
    if not midi_files:
        print(f"[Aviso] Nenhum arquivo .mid em: {folder}")
        return

    print(f"🎧 Sintetizando {len(midi_files)} arquivos MIDI em áudio com arranjo e ambiência...")
    for m in midi_files:
        convert_midi_to_audio(m, output_dir=folder, output_format="mp3")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            convert_all_in_folder(target)
        else:
            convert_midi_to_audio(target)
    else:
        convert_all_in_folder("output")
