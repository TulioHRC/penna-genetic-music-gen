"""
Audio to MIDI Transcriber & Downloader:
- Faz download direto de links do YouTube (áudio) usando yt-dlp.
- Transcreve áudio (.mp3, .wav, .m4a, .mp4) para MIDI (.mid) usando detecção de pitch e onsets via Librosa / PrettyMIDI.
"""

import os
import re
from typing import Optional
import numpy as np
import librosa
import pretty_midi
import yt_dlp


class AudioToMidiTranscriber:
    """Extrai notas melódicas e ritmo de um arquivo de áudio para gerar um MIDI limpo."""

    @staticmethod
    def download_youtube_audio(url: str, output_dir: str = "data/reference_midi") -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_tmpl = os.path.join(output_dir, "%(title).50s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }] if False else [],  # yt-dlp baixa em m4a/webm nativo e librosa lê direto
            "quiet": False,
            "no_warnings": True,
        }

        print(f"📥 Baixando áudio do YouTube: {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # Se for vídeo ou outro formato, librosa consegue ler diretamente a faixa de áudio
            return filename

    @staticmethod
    def transcribe_audio_to_midi(
        audio_path: str,
        output_midi_path: Optional[str] = None,
        fmin: float = librosa.note_to_hz("C2"),
        fmax: float = librosa.note_to_hz("C6"),
        hop_length: int = 512,
        energy_threshold: float = 0.08
    ) -> str:
        """
        Transcreve a melodia/linha principal de um áudio para MIDI
        utilizando algoritmo de tracking de pitch (pYIN / tracking fundamental) do Librosa.
        """
        print(f"🎵 Carregando áudio para transcrição: {audio_path}...")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        if output_midi_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_midi_path = os.path.join("data/reference_midi", f"{base_name}.mid")

        print("🔍 Estimando frequências e notas melódicas (pYIN)...")
        # Estimação de pitch com pYIN (rápido e robusto para melodias/solos/voz)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_norm = rms / (np.max(rms) + 1e-6)

        pm = pretty_midi.PrettyMIDI()
        guitar = pretty_midi.Instrument(program=29, name="Transcribed Melody")

        current_pitch = None
        start_time = 0.0

        for t, freq, voiced, v_prob, amp in zip(times, f0, voiced_flag, voiced_probs, rms_norm):
            if voiced and not np.isnan(freq) and amp > energy_threshold and v_prob > 0.4:
                midi_note = int(round(librosa.hz_to_midi(freq)))
                if current_pitch is None:
                    current_pitch = midi_note
                    start_time = t
                elif abs(midi_note - current_pitch) >= 1:  # Mudança de nota
                    if t - start_time >= 0.08:  # Mínimo de 80ms para evitar ruído transitório
                        note = pretty_midi.Note(
                            velocity=int(np.clip(amp * 127, 60, 115)),
                            pitch=current_pitch,
                            start=float(start_time),
                            end=float(t)
                        )
                        guitar.notes.append(note)
                    current_pitch = midi_note
                    start_time = t
            else:
                if current_pitch is not None:
                    if t - start_time >= 0.08:
                        note = pretty_midi.Note(
                            velocity=90,
                            pitch=current_pitch,
                            start=float(start_time),
                            end=float(t)
                        )
                        guitar.notes.append(note)
                    current_pitch = None

        if current_pitch is not None and len(times) > 0:
            guitar.notes.append(pretty_midi.Note(
                velocity=90,
                pitch=current_pitch,
                start=float(start_time),
                end=float(times[-1])
            ))

        pm.instruments.append(guitar)
        os.makedirs(os.path.dirname(output_midi_path) or ".", exist_ok=True)
        pm.write(output_midi_path)
        print(f"✨ Transcrição concluída! Extraídas {len(guitar.notes)} notas -> {output_midi_path}")
        return output_midi_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        if inp.startswith("http://") or inp.startswith("https://"):
            aud = AudioToMidiTranscriber.download_youtube_audio(inp)
            AudioToMidiTranscriber.transcribe_audio_to_midi(aud)
        else:
            AudioToMidiTranscriber.transcribe_audio_to_midi(inp)
