"""
Audio to MIDI Transcriber & Downloader:
- Faz download direto de links do YouTube (áudio) usando yt-dlp.
- Converte automaticamente formatos baixados (.webm, .m4a, .mp4, etc.) para WAV via ffmpeg embutido (imageio-ffmpeg).
- Transcreve áudio para MIDI (.mid) usando detecção de pitch robusta (YIN) e onsets do Librosa.
"""

import os
import subprocess
from typing import Optional
import soundfile as sf
import numpy as np
import librosa
import pretty_midi
import yt_dlp
import imageio_ffmpeg


# WAV em qualidade de CD e estéreo: os samples extraídos precisam dos agudos (chimbal)
# e da imagem estéreo (lead sem voz vem do canal lateral L-R, ver sample_kit.py)
WAV_SAMPLE_RATE = 44100
WAV_CHANNELS = 2
ORIGINAL_AUDIO_EXTENSIONS = (".webm", ".m4a", ".mp3", ".opus", ".ogg", ".mp4", ".flac")


class AudioToMidiTranscriber:
    """Extrai notas melódicas e ritmo de um arquivo de áudio para gerar um MIDI limpo."""

    @staticmethod
    def _convert_to_wav_if_needed(audio_path: str) -> str:
        """Garante que o áudio seja um arquivo WAV legível pelo librosa/soundfile."""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == ".wav":
            return audio_path

        wav_path = os.path.splitext(audio_path)[0] + ".wav"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        print(f"🔄 Convertendo áudio ({ext}) para WAV via FFmpeg...")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", audio_path,
            "-vn",
            "-ar", str(WAV_SAMPLE_RATE),
            "-ac", str(WAV_CHANNELS),
            wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(wav_path):
            return wav_path
        return audio_path

    @staticmethod
    def best_wav_for(base_path: str) -> Optional[str]:
        """
        WAV de uma referência (caminho sem extensão). Se o WAV existente for de uma versão
        antiga (mono / 22 kHz) e o áudio original ainda existir, converte de novo.
        """
        wav_path = base_path + ".wav"
        originals = [base_path + ext for ext in ORIGINAL_AUDIO_EXTENSIONS if os.path.exists(base_path + ext)]
        if os.path.exists(wav_path):
            info = sf.info(wav_path)
            is_outdated = info.channels < WAV_CHANNELS or info.samplerate < WAV_SAMPLE_RATE
            if not (is_outdated and originals):
                return wav_path
        if originals:
            return AudioToMidiTranscriber._convert_to_wav_if_needed(originals[0])
        return None

    @staticmethod
    def download_youtube_audio(
        url: str,
        output_dir: str = "data/reference_midi",
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str] = None
    ) -> str:
        """
        Baixa o áudio de um vídeo do YouTube.
        Se o YouTube pedir login ("confirm you're not a bot"), passe cookies:
        - cookies_from_browser: "BROWSER[:PERFIL]" (ex.: "firefox" ou "chrome:/caminho/do/perfil")
        - cookies_file: arquivo cookies.txt no formato Netscape
        """
        os.makedirs(output_dir, exist_ok=True)
        out_tmpl = os.path.join(output_dir, "%(title).50s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "quiet": False,
            # YouTube exige resolver um desafio JavaScript; habilita qualquer runtime instalado
            # (o padrão do yt-dlp é só deno). Os scripts do solver vêm do pacote yt-dlp-ejs.
            "js_runtimes": {"deno": {}, "node": {}, "bun": {}},
        }
        if cookies_from_browser:
            browser, _, profile = cookies_from_browser.partition(":")
            ydl_opts["cookiesfrombrowser"] = (browser, profile or None, None, None)
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file

        print(f"📥 Baixando áudio do YouTube: {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        # Converte para WAV caso venha em .webm, .opus, .m4a
        wav_file = AudioToMidiTranscriber._convert_to_wav_if_needed(filename)
        return wav_file

    @staticmethod
    def transcribe_audio_to_midi(
        audio_path: str,
        output_midi_path: Optional[str] = None,
        fmin: float = librosa.note_to_hz("C2"),
        fmax: float = librosa.note_to_hz("C6"),
        hop_length: int = 512,
        energy_threshold: float = 0.05
    ) -> str:
        """
        Transcreve a melodia/linha principal de um áudio para MIDI
        utilizando algoritmo de tracking de pitch (YIN) e onsets do Librosa.
        """
        safe_audio_path = AudioToMidiTranscriber._convert_to_wav_if_needed(audio_path)

        print(f"🎵 Carregando áudio para transcrição: {safe_audio_path}...")
        y, sr = librosa.load(safe_audio_path, sr=22050, mono=True)

        if output_midi_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_midi_path = os.path.join("data/reference_midi", f"{base_name}.mid")

        print("🔍 Estimando frequências e notas melódicas (YIN)...")
        f0 = librosa.yin(
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

        for t, freq, amp in zip(times, f0, rms_norm):
            is_voiced = (amp > energy_threshold) and (not np.isnan(freq)) and (freq >= fmin) and (freq <= fmax)

            if is_voiced:
                midi_note = int(round(librosa.hz_to_midi(freq)))
                if current_pitch is None:
                    current_pitch = midi_note
                    start_time = t
                elif abs(midi_note - current_pitch) >= 1:
                    if t - start_time >= 0.09:
                        note = pretty_midi.Note(
                            velocity=int(np.clip(amp * 127, 65, 115)),
                            pitch=current_pitch,
                            start=float(start_time),
                            end=float(t)
                        )
                        guitar.notes.append(note)
                    current_pitch = midi_note
                    start_time = t
            else:
                if current_pitch is not None:
                    if t - start_time >= 0.09:
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
