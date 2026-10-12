"""
Advanced Realistic Audio Synthesizer:
1. Renderiza os instrumentos MIDI do arranjo usando síntese aditiva e física especializada:
   - Guitarra (Lead Guitar): Modelagem de captador, harmônicos saturados (overdrive valvulado) e ressonância de corda.
   - Bateria Real (Rock Drum Kit):
       • Bumbo (Kick): Transiente de impacto e queda senoidal rápida sub-grave (160Hz -> 45Hz).
       • Caixa (Snare): Tom de corpo acoplado a ruído de esteira metálica.
       • Chimbal (Hi-Hat) e Prato (Crash): Ruído filtrado passa-alta metálico de decaimento rápido.
   - Baixo Elétrico (Electric Bass): Ataque percussivo de palheta/dedo e fundamental encorpada.
   - Trompete / Metais (Brass): Harmônicos brilhantes de sopro e ataque labial.
2. Mixagem multi-pistas com espacialização estéreo e ambiência de sala.
3. Exportação automática tanto em MIDI, MusicXML (.musicxml) e Áudio (.mp3 / .wav).
"""

import os
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import soundfile as sf
import pretty_midi

from sample_renderer import render_with_samples
from soundfont_renderer import render_with_soundfont, soundfont_available


class RealisticBandSynthesizer:
    """Sintetizador especializado em timbres acústicos e de rock para instrumentos MIDI."""

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def _render_guitar_note(self, freq: float, duration: float, velocity: float) -> np.ndarray:
        """Sintetiza uma nota de guitarra elétrica com overdrive e ressonância de corda."""
        n_samples = max(int(self.sr * duration), 1)
        t = np.linspace(0, duration, n_samples, False)

        # Harmônicos ricos de corda metálica
        harmonic_1 = np.sin(2 * np.pi * freq * t)
        harmonic_2 = 0.5 * np.sin(2 * np.pi * 2 * freq * t)
        harmonic_3 = 0.25 * np.sin(2 * np.pi * 3 * freq * t)
        raw_wave = harmonic_1 + harmonic_2 + harmonic_3

        # Envelope ADSR de palhetada
        attack_len = min(int(self.sr * 0.015), n_samples)
        env = np.ones(n_samples)
        if attack_len > 0:
            env[:attack_len] = np.linspace(0, 1, attack_len)
        decay = np.exp(-t * (1.8 + (freq / 400.0)))
        env = env * decay

        # Modelagem de captador (ressonância em ~2.5kHz)
        b, a = signal.butter(2, [max(80, freq * 0.8), min(3800, self.sr * 0.45)], btype='bandpass', fs=self.sr)
        filtered = signal.lfilter(b, a, raw_wave * env)

        # Distorção/Overdrive valvulado de amplificador de guitarra
        drive = 3.2 * (velocity / 100.0)
        distorted = np.tanh(drive * filtered)
        return distorted * (velocity / 127.0)

    def _render_bass_note(self, freq: float, duration: float, velocity: float) -> np.ndarray:
        """Sintetiza uma nota de baixo elétrico encorpada e definida."""
        n_samples = max(int(self.sr * duration), 1)
        t = np.linspace(0, duration, n_samples, False)

        # Fundamental forte + segundo harmônico (peso e definição)
        fund = np.sin(2 * np.pi * freq * t)
        harm = 0.4 * np.sin(2 * np.pi * 2 * freq * t)
        attack_click = np.random.uniform(-0.1, 0.1, n_samples) * np.exp(-t * 80)

        env = np.exp(-t * 2.2)
        bass_sound = (fund + harm + attack_click) * env
        b, a = signal.butter(2, [35, min(1200, self.sr * 0.45)], btype='bandpass', fs=self.sr)
        return signal.lfilter(b, a, bass_sound) * (velocity / 127.0)

    def _render_brass_note(self, freq: float, duration: float, velocity: float) -> np.ndarray:
        """Sintetiza metais/trompete com ataque e riqueza harmônica."""
        n_samples = max(int(self.sr * duration), 1)
        t = np.linspace(0, duration, n_samples, False)

        # Série harmônica de sopro de metal
        wave = (
            np.sin(2 * np.pi * freq * t) +
            0.6 * np.sin(2 * np.pi * 2 * freq * t) +
            0.35 * np.sin(2 * np.pi * 3 * freq * t) +
            0.15 * np.sin(2 * np.pi * 4 * freq * t)
        )
        attack_len = min(int(self.sr * 0.04), n_samples)
        env = np.ones(n_samples)
        if attack_len > 0:
            env[:attack_len] = np.sin(np.linspace(0, np.pi / 2, attack_len))
        env[-min(int(self.sr * 0.05), n_samples):] *= np.linspace(1, 0, min(int(self.sr * 0.05), n_samples))

        b, a = signal.butter(2, [max(120, freq * 0.8), min(4500, self.sr * 0.45)], btype='bandpass', fs=self.sr)
        return signal.lfilter(b, a, wave * env) * (velocity / 127.0)

    def _render_drum_hit(self, pitch: int, velocity: float) -> np.ndarray:
        """Sintetiza batidas reais de bateria rock acústica."""
        # Bumbo (35 ou 36)
        if pitch in (35, 36):
            dur = 0.4
            n_samples = int(self.sr * dur)
            t = np.linspace(0, dur, n_samples, False)
            f_env = 45 + 120 * np.exp(-t * 38)
            phase = 2 * np.pi * np.cumsum(f_env) / self.sr
            click = np.random.uniform(-0.3, 0.3, n_samples) * np.exp(-t * 90)
            kick = (np.sin(phase) + click) * np.exp(-t * 9.5)
            return np.tanh(2.5 * kick) * (velocity / 127.0)

        # Caixa (38 ou 40)
        elif pitch in (38, 39, 40):
            dur = 0.35
            n_samples = int(self.sr * dur)
            t = np.linspace(0, dur, n_samples, False)
            tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 18)
            noise = np.random.uniform(-1, 1, n_samples)
            b, a = signal.butter(2, [900, 7500], btype='bandpass', fs=self.sr)
            snare_noise = signal.lfilter(b, a, noise) * np.exp(-t * 11)
            snare = (tone * 0.35 + snare_noise * 0.75)
            return np.tanh(2.0 * snare) * (velocity / 127.0)

        # Chimbal fechado (42 ou 44)
        elif pitch in (42, 44):
            dur = 0.09
            n_samples = int(self.sr * dur)
            t = np.linspace(0, dur, n_samples, False)
            noise = np.random.uniform(-1, 1, n_samples)
            b, a = signal.butter(2, [6500, 15000], btype='bandpass', fs=self.sr)
            hh = signal.lfilter(b, a, noise) * np.exp(-t * 50)
            return hh * (velocity / 127.0) * 0.65

        # Prato de ataque / Crash (49 ou 57) ou Chimbal aberto (46)
        elif pitch in (46, 49, 57):
            dur = 0.85
            n_samples = int(self.sr * dur)
            t = np.linspace(0, dur, n_samples, False)
            noise = np.random.uniform(-1, 1, n_samples)
            b, a = signal.butter(2, [5000, 14000], btype='bandpass', fs=self.sr)
            crash = signal.lfilter(b, a, noise) * np.exp(-t * 4.0)
            return crash * (velocity / 127.0) * 0.55

        else:
            # Tons (low/mid/high) e percussão genérica
            dur = 0.25
            n_samples = int(self.sr * dur)
            t = np.linspace(0, dur, n_samples, False)
            tom_freqs = {41: 90.0, 43: 100.0, 45: 110.0, 47: 140.0, 48: 165.0, 50: 190.0}
            f = tom_freqs.get(pitch, 120.0)
            return np.sin(2 * np.pi * f * t) * np.exp(-t * 15) * (velocity / 127.0)

    def synthesize_midi(self, pm: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Renderiza todo o PrettyMIDI com mixagem dos instrumentos."""
        total_time = pm.get_end_time() + 1.2
        total_samples = int(total_time * self.sr)
        mix = np.zeros(total_samples, dtype=np.float32)

        for inst in pm.instruments:
            track = np.zeros(total_samples, dtype=np.float32)

            for note in inst.notes:
                start_sample = int(note.start * self.sr)
                dur = max(0.04, note.end - note.start)

                if inst.is_drum:
                    sample_audio = self._render_drum_hit(note.pitch, note.velocity)
                elif "guitar" in inst.name.lower() or inst.program in range(25, 32):
                    freq = pretty_midi.note_number_to_hz(note.pitch)
                    sample_audio = self._render_guitar_note(freq, dur, note.velocity)
                elif "bass" in inst.name.lower() or inst.program in range(32, 40):
                    freq = pretty_midi.note_number_to_hz(note.pitch)
                    sample_audio = self._render_bass_note(freq, dur, note.velocity)
                elif inst.program in range(48, 64) or any(w in inst.name.lower() for w in ("trumpet", "brass", "string", "pad")):
                    freq = pretty_midi.note_number_to_hz(note.pitch)
                    sample_audio = self._render_brass_note(freq, dur, note.velocity)
                else:
                    freq = pretty_midi.note_number_to_hz(note.pitch)
                    sample_audio = self._render_guitar_note(freq, dur, note.velocity)

                end_sample = min(total_samples, start_sample + len(sample_audio))
                length = end_sample - start_sample
                if length > 0:
                    track[start_sample:end_sample] += sample_audio[:length]

            # Balanço de volume na mixagem
            if inst.is_drum:
                mix += track * 0.95
            elif "bass" in inst.name.lower():
                mix += track * 0.85
            elif inst.program in range(48, 64):
                mix += track * 0.65
            else:
                mix += track * 0.80

        # Ambiência de sala (Reverberação suave)
        delays = [int(self.sr * 0.027), int(self.sr * 0.038), int(self.sr * 0.052)]
        gains = [0.20, 0.14, 0.09]
        reverb = np.zeros_like(mix)
        for d, g in zip(delays, gains):
            if len(mix) > d:
                reverb[d:] += g * mix[:-d]

        master = 0.85 * mix + 0.25 * reverb
        peak = np.max(np.abs(master)) + 1e-6
        return (master / peak) * 0.93


def convert_midi_to_audio(
    midi_path: str,
    output_dir: str = None,
    output_format: str = "mp3",
    samplerate: int = 44100,
    soundfont: str = None,
    sample_kit=None
) -> str:
    """
    Converte MIDI para WAV e MP3. Fonte de timbre, em ordem de preferência:
    1. sample_kit: sons recortados da música de referência (SoundFont completa o que faltar);
    2. SoundFont: samples de instrumentos General MIDI;
    3. sintetizador aditivo (som mais artificial).
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"Arquivo MIDI não encontrado: {midi_path}")

    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    target_dir = output_dir if output_dir else os.path.dirname(midi_path)
    os.makedirs(target_dir, exist_ok=True)

    wav_out = os.path.join(target_dir, f"{base_name}.wav")
    mp3_out = os.path.join(target_dir, f"{base_name}.mp3")

    pm = pretty_midi.PrettyMIDI(midi_path)
    has_soundfont = bool(soundfont) and os.path.exists(soundfont) and soundfont_available()
    if sample_kit is not None:
        audio = render_with_samples(pm, sample_kit, soundfont if has_soundfont else None, sr=samplerate)
        engine = "sons da referência"
    elif has_soundfont:
        audio = render_with_soundfont(pm, soundfont, sr=samplerate)
        engine = f"SoundFont {os.path.basename(soundfont)}"
    else:
        if soundfont:
            print(f"  [Aviso] SoundFont indisponível ('{soundfont}' ou pacote tinysoundfont ausente); "
                  "usando sintetizador aditivo (som mais artificial).")
        audio = RealisticBandSynthesizer(sr=samplerate).synthesize_midi(pm)
        engine = "sintetizador aditivo"

    # Grava WAV (PCM 16-bit)
    wavfile.write(wav_out, samplerate, (audio * 32767).astype(np.int16))

    # Codifica MP3
    data, sr = sf.read(wav_out)
    sf.write(mp3_out, data, sr, format="MP3")

    # Também exporta MusicXML para compatibilidade com MuseScore / Sibelius se music21 estiver disponível
    try:
        import music21
        score = music21.converter.parse(midi_path)
        musicxml_out = os.path.join(target_dir, f"{base_name}.musicxml")
        score.write("musicxml", musicxml_out)
    except Exception:
        pass

    print(f"  🔊 Áudio gerado com {engine}: {mp3_out} (WAV: {wav_out})")
    return mp3_out if output_format.lower() == "mp3" else wav_out


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "output/penna_generated_1.mid"
    convert_midi_to_audio(target, soundfont=sys.argv[2] if len(sys.argv) > 2 else None)
