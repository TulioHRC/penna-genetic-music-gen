#!/usr/bin/env python3
"""
CLI Principal: Penna Genetic Music Generator
Permite passar:
1. Links do YouTube (--youtube "https://...") -> baixa áudio e transcreve para MIDI.
2. Arquivos de áudio (.mp3, .wav, .m4a, .mp4) (--audio musica.mp3) -> transcreve para MIDI.
3. Arquivos MIDI diretos (--references data/*.mid).

Extrai padrões de estilo para a função de fitness e evolui novas músicas via Algoritmo Genético.
"""

import argparse
import glob
import os
import sys

from corpus_analyzer import CorpusAnalyzer
from fitness_evaluator import CorpusFitnessEvaluator
from genetic_engine import GeneticEngine
from demo_references import create_pitty_riff_demo
from audio_transcriber import AudioToMidiTranscriber
from audio_synth import convert_midi_to_audio


def main():
    parser = argparse.ArgumentParser(
        description="Penna: Geração de Música Simbólica via Algoritmos Genéticos com Estilo Musical (Pitty / Rock)"
    )
    # Entradas de Referência
    parser.add_argument(
        "--youtube",
        nargs="+",
        default=[],
        help="Links do YouTube para baixar, transcrever para MIDI e usar de referência"
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=[],
        help="Arquivos de áudio (.mp3, .wav, .mp4) para transcrever para MIDI e usar de referência"
    )
    parser.add_argument(
        "--references",
        nargs="+",
        default=[],
        help="Arquivos MIDI (.mid) existentes de referência"
    )

    # Hiperparâmetros do Algoritmo Genético
    parser.add_argument("--bars", type=int, default=16, help="Quantidade de compassos a gerar (default: 16 compassos = ~32s a 120bpm; use 16 para ~30s ou 32 para ~60s)")
    parser.add_argument("--generations", type=int, default=80, help="Número de gerações (default: 80)")
    parser.add_argument("--population", type=int, default=60, help="Tamanho da população (default: 60)")
    parser.add_argument("--mutation-rate", type=float, default=0.20, help="Taxa de mutação (default: 0.20)")
    parser.add_argument("--crossover-rate", type=float, default=0.80, help="Taxa de crossover (default: 0.80)")
    parser.add_argument("--bpm", type=float, default=120.0, help="Andamento da música em BPM (default: 120.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed pseudo-aleatória")
    parser.add_argument("--num-outputs", type=int, default=3, help="Quantidade de músicas a gerar (default: 3)")
    parser.add_argument("--output-dir", type=str, default="output", help="Diretório de saída dos resultados")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/reference_midi", exist_ok=True)

    ref_files = []

    # 1. Processar links do YouTube
    if args.youtube:
        print("\n🌐 Processando links do YouTube...")
        for url in args.youtube:
            try:
                audio_file = AudioToMidiTranscriber.download_youtube_audio(url)
                midi_file = AudioToMidiTranscriber.transcribe_audio_to_midi(audio_file)
                ref_files.append(midi_file)
            except Exception as e:
                print(f"[Erro] Falha ao processar YouTube '{url}': {e}")

    # 2. Processar arquivos de áudio (.mp3, .wav, .mp4)
    if args.audio:
        print("\n🎙️ Transcrevendo arquivos de áudio para MIDI...")
        for audio_path in args.audio:
            for matched in glob.glob(audio_path) if any(c in audio_path for c in "*?[]") else [audio_path]:
                if os.path.exists(matched):
                    try:
                        midi_file = AudioToMidiTranscriber.transcribe_audio_to_midi(matched)
                        ref_files.append(midi_file)
                    except Exception as e:
                        print(f"[Erro] Falha ao transcrever áudio '{matched}': {e}")

    # 3. Processar arquivos MIDI existentes
    if args.references:
        for ref_arg in args.references:
            matches = glob.glob(ref_arg)
            if matches:
                ref_files.extend(matches)
            elif os.path.exists(ref_arg):
                ref_files.append(ref_arg)

    # Fallback se nenhuma entrada foi informada
    if not ref_files:
        existing_in_data = glob.glob("data/reference_midi/*.mid")
        if existing_in_data:
            ref_files = existing_in_data
        else:
            print("[Aviso] Nenhuma referência encontrada. Criando riffs de demonstração estilo Pitty...")
            ref1 = "data/reference_midi/pitty_mascara_demo.mid"
            ref2 = "data/reference_midi/pitty_chipnovo_demo.mid"
            create_pitty_riff_demo(ref1, "Mascara")
            create_pitty_riff_demo(ref2, "Chip Novo")
            ref_files = [ref1, ref2]

    # Elimina duplicatas preservando ordem
    ref_files = list(dict.fromkeys(ref_files))

    print(f"\n🎸 [1/4] Analisando corpus de referência ({len(ref_files)} arquivos):")
    for r in ref_files:
        print(f"  -> {r}")

    analyzer = CorpusAnalyzer(ref_files)
    profile = analyzer.build_profile()

    print("\n📊 Perfil de Estilo Extraído:")
    print(f"  • Tessitura: Min {profile.min_pitch} | Max {profile.max_pitch} | Média {profile.avg_pitch:.1f}")
    top_pcs = [str(pc) for pc, _ in sorted(enumerate(profile.pitch_class_distribution), key=lambda x: x[1], reverse=True)[:5]]
    print(f"  • Top Pitch Classes mais frequentes: {', '.join(top_pcs)}")
    print(f"  • Transições de Intervalos Melódicos mapeadas: {len(profile.interval_transition_distribution)} transições")

    evaluator = CorpusFitnessEvaluator(
        profile=profile,
        w_pitch_class=0.35,
        w_interval=0.35,
        w_register=0.15,
        w_rhythm=0.15
    )

    print(f"\n🧬 [2/4] Executando Algoritmo Genético para gerar {args.num_outputs} variações musicais...")

    for song_idx in range(1, args.num_outputs + 1):
        current_seed = args.seed + (song_idx * 17)
        current_mut = args.mutation_rate + (song_idx - 1) * 0.05

        print(f"\n--- Gerando Peça #{song_idx} (Seed={current_seed}, Taxa Mutação={current_mut:.2f}) ---")
        steps_to_generate = args.bars * 16  # 16 steps (semicolcheias) por compasso em 4/4
        engine = GeneticEngine(
            evaluator=evaluator,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=current_mut,
            crossover_rate=args.crossover_rate,
            random_seed=current_seed,
            total_steps=steps_to_generate
        )

        best_ind, history_best, history_avg = engine.run()

        # 1. Salva o MIDI com arranjo de banda completa (Guitarra Solo, Trompete, Baixo e Bateria de Rock)
        from music_arranger import MusicArranger
        band_pm = MusicArranger.create_full_band_midi(
            melody_individual=best_ind,
            bpm=args.bpm,
            include_drums=True,
            include_bass=True,
            include_brass=True
        )
        midi_filename = os.path.join(args.output_dir, f"penna_generated_{song_idx}.mid")
        band_pm.write(midi_filename)
        print(f"  ✓ MIDI (Banda Completa: Guitarra + Trompete + Baixo + Bateria): {midi_filename} (Fitness: {best_ind.fitness:.4f})")

        # 2. Sintetiza em MP3 e WAV com ambiência de sala e saturação analógica
        try:
            convert_midi_to_audio(midi_filename, output_dir=args.output_dir, output_format="mp3")
        except Exception as e:
            print(f"  [Aviso] Falha na síntese de áudio: {e}")

        # 3. Salva gráfico de convergência do fitness
        plot_filename = os.path.join(args.output_dir, f"fitness_evolution_{song_idx}.png")
        engine.plot_fitness_history(plot_filename)

    print(f"\n✨ [3/4] Concluído com sucesso! Arquivos MIDI, MP3, WAV e gráficos salvos em '{args.output_dir}'.")
    print(f"🎧 [4/4] Áudios sintetizados e prontos para reprodução.")


if __name__ == "__main__":
    main()
