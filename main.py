"""
CLI Principal: Penna Genetic Music Generator
Permite passar:
1. Links do YouTube (--youtube "https://...") -> baixa áudio e transcreve para MIDI.
2. Arquivos de áudio (.mp3, .wav, .m4a, .mp4) (--audio musica.mp3) -> transcreve para MIDI.
3. Arquivos MIDI diretos (--references data/*.mid).
Sem argumentos (`python main.py`), abre um formulário interativo (tui.py) com os valores padrão.

Extrai padrões de estilo para a função de fitness e evolui novas músicas via Algoritmo Genético.
"""

import argparse
import glob
import os
import random
import sys
from typing import List, Optional
from corpus_analyzer import CorpusAnalyzer
from alg_genetic_fitness_evaluator import CorpusFitnessEvaluator
from alg_genetic_engine import GeneticEngine
from audio_transcriber import AudioToMidiTranscriber
from audio_synth import convert_midi_to_audio
from music_arranger import GROOVE_HIT_THRESHOLD, MusicArranger, detect_key
from groove_extractor import Groove, average_grooves, extract_groove
from styles import STYLES
from sample_kit import SampleKit, extract_sample_kit
from config import load_defaults

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def build_parser() -> argparse.ArgumentParser:
    """Define todas as opções. Os grupos viram as seções do formulário interativo (tui.py)."""
    defaults = load_defaults()
    parser = argparse.ArgumentParser(
        description="Penna: Geração de Música Simbólica via Algoritmos Genéticos com Estilo Musical (Pitty / Rock)"
    )

    references = parser.add_argument_group("Referências")
    references.add_argument(
        "--youtube",
        nargs="*",
        default=defaults["youtube"],
        help="Links do YouTube para baixar, transcrever para MIDI e usar de referência"
    )
    references.add_argument(
        "--cookies-from-browser",
        default=defaults["cookies_from_browser"],
        help="Usa os cookies do navegador para o YouTube: BROWSER[:PERFIL] (ex.: firefox, chrome)"
    )
    references.add_argument("--cookies", default=defaults["cookies"], help="Arquivo cookies.txt (formato Netscape) para o YouTube")
    references.add_argument(
        "--audio",
        nargs="*",
        default=defaults["audio"],
        help="Arquivos de áudio (.mp3, .wav, .mp4) para transcrever para MIDI e usar de referência"
    )
    references.add_argument(
        "--references",
        nargs="*",
        default=defaults["references"],
        help="Arquivos MIDI (.mid) existentes de referência"
    )

    genetic = parser.add_argument_group("Algoritmo Genético")
    genetic.add_argument("--bars", type=int, default=defaults["bars"], help="Quantidade de compassos a gerar (16 ≈ 30s, 32 ≈ 60s)")
    genetic.add_argument("--generations", type=int, default=defaults["generations"], help="Número de gerações")
    genetic.add_argument("--population", type=int, default=defaults["population"], help="Tamanho da população")
    genetic.add_argument("--mutation-rate", type=float, default=defaults["mutation_rate"], help="Taxa de mutação")
    genetic.add_argument("--crossover-rate", type=float, default=defaults["crossover_rate"], help="Taxa de crossover")
    genetic.add_argument("--seed", type=int, default=defaults["seed"], help="Seed pseudo-aleatória (vazio: aleatória a cada execução)")
    genetic.add_argument("--num-outputs", type=int, default=defaults["num_outputs"], help="Quantidade de músicas a gerar")

    arrangement = parser.add_argument_group("Arranjo e Saída")
    arrangement.add_argument("--style", choices=sorted(STYLES), default=defaults["style"], help="Estilo do arranjo: timbres, kit de bateria e papel de cada instrumento")
    arrangement.add_argument("--bpm", type=float, default=defaults["bpm"], help="Andamento em BPM (vazio: andamento detectado no áudio de referência)")
    arrangement.add_argument("--no-drums", action="store_true", default=defaults["no_drums"], help="Não gera a pista de bateria")
    arrangement.add_argument("--no-bass", action="store_true", default=defaults["no_bass"], help="Não gera a pista de baixo")
    arrangement.add_argument("--no-harmony", action="store_true", default=defaults["no_harmony"], help="Não gera a pista de acordes (cordas)")
    arrangement.add_argument("--timbre", choices=["reference", "soundfont"], default=defaults["timbre"], help="reference: sons recortados da música de referência | soundfont: instrumentos General MIDI")
    arrangement.add_argument("--soundfont", default=defaults["soundfont"], help="Arquivo .sf2 com samples reais de instrumentos (vazio: sintetizador aditivo)")
    arrangement.add_argument("--output-dir", type=str, default=defaults["output_dir"], help="Diretório de saída dos resultados")

    # Pesos relativos: o avaliador divide cada um pela soma, então só a proporção importa
    weights = parser.add_argument_group("Pesos do Fitness (só a proporção entre eles importa)")
    weights.add_argument("--w-pitch", type=float, default=defaults["w_pitch"], help="Usar as notas mais frequentes da referência. Suba se a melodia sair do tom")
    weights.add_argument("--w-interval", type=float, default=defaults["w_interval"], help="Saltos entre notas parecidos com os da referência; pune repetição e saltos enormes")
    weights.add_argument("--w-motif", type=float, default=defaults["w_motif"], help="Repetir pequenos motivos (20–55% de repetição). Suba se soar aleatório")
    weights.add_argument("--w-resolution", type=float, default=defaults["w_resolution"], help="Terminar na tônica (ou quinta/terça). Só olha a última nota")
    weights.add_argument("--w-register", type=float, default=defaults["w_register"], help="Ficar na mesma região (grave/agudo) da referência")
    weights.add_argument("--w-rhythm", type=float, default=defaults["w_rhythm"], help="Ter de 10% a 35% de pausas, para a melodia respirar")
    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        # Sem argumentos: abre o formulário interativo
        from tui import edit_arguments
        args = edit_arguments(parser)
        if args is None:
            return
    else:
        args = parser.parse_args()
    run(args)


def _expand_path(path: str) -> str:
    """Expande ~ e $VARIAVEIS, já que valores vindos do config.json/formulário não passam pelo shell."""
    return os.path.expandvars(os.path.expanduser(path))


def _reference_audio_for(midi_path: str) -> Optional[str]:
    """Áudio original de um MIDI transcrito (o transcritor salva o áudio ao lado do .mid)."""
    return AudioToMidiTranscriber.best_wav_for(os.path.splitext(midi_path)[0])


def _build_groove(audio_paths: List[str], fallback: Groove) -> Groove:
    """Extrai o groove dos áudios de referência; sem áudio, usa o groove padrão do estilo."""
    grooves = []
    for path in dict.fromkeys(audio_paths):
        try:
            grooves.append(extract_groove(path))
        except Exception as e:
            print(f"  [Aviso] Não foi possível extrair o groove de '{path}': {e}")
    return average_grooves(grooves) if grooves else fallback


def _build_sample_kit(audio_paths: List[str], output_dir: str) -> Optional[SampleKit]:
    """Recorta os sons da primeira referência com áudio e salva os samples para conferência."""
    for path in dict.fromkeys(audio_paths):
        try:
            kit = extract_sample_kit(path)
        except Exception as e:
            print(f"  [Aviso] Não foi possível extrair sons de '{path}': {e}")
            continue
        kit_dir = os.path.join(output_dir, "sample_kit")
        kit.save(kit_dir)
        print(f"  • Sons extraídos de '{os.path.basename(path)}': {kit.describe()} (salvos em {kit_dir})")
        return kit
    return None


FITNESS_CRITERIA_LABELS = {
    "pitch": "Notas/escala (w_pitch)",
    "interval": "Intervalos (w_interval)",
    "motif": "Motivos (w_motif)",
    "resolution": "Resolução final (w_resolution)",
    "register": "Tessitura (w_register)",
    "rhythm": "Pausas (w_rhythm)",
}
SCORE_BAR_WIDTH = 20


def _print_fitness_breakdown(evaluator: CorpusFitnessEvaluator, individual) -> None:
    """Mostra a nota de cada critério e quanto ele somou no fitness, para guiar o ajuste dos pesos."""
    scores = evaluator.score_breakdown(individual)
    if scores is None:
        return
    print("    Critério                          nota          peso   contribuição")
    for name, score in scores.items():
        weight = evaluator.weights[name]
        filled = int(round(score * SCORE_BAR_WIDTH))
        bar = "█" * filled + "░" * (SCORE_BAR_WIDTH - filled)
        hint = "  ← já no máximo: aumentar o peso não muda nada" if score >= 0.999 else ""
        print(f"    {FITNESS_CRITERIA_LABELS[name]:<32} {bar} {score:4.2f}  {weight:4.0%}  {weight * score:5.3f}{hint}")


def run(args: argparse.Namespace) -> None:
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    args.audio = [_expand_path(p) for p in args.audio]
    args.references = [_expand_path(p) for p in args.references]
    args.cookies = _expand_path(args.cookies) if args.cookies else None
    args.output_dir = _expand_path(args.output_dir)
    args.soundfont = _expand_path(args.soundfont) if args.soundfont else None
    print(f"🎲 Seed: {args.seed}")

    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/reference_midi", exist_ok=True)

    ref_files = []
    ref_audio = []  # áudios originais, usados para extrair groove e andamento

    # 1. Processar links do YouTube
    if args.youtube:
        print("\n🌐 Processando links do YouTube...")
        for url in args.youtube:
            try:
                audio_file = AudioToMidiTranscriber.download_youtube_audio(
                    url, cookies_from_browser=args.cookies_from_browser, cookies_file=args.cookies
                )
                midi_file = AudioToMidiTranscriber.transcribe_audio_to_midi(audio_file)
                ref_files.append(midi_file)
                ref_audio.append(audio_file)
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
                        ref_audio.append(matched)
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
        print("\n⚠️ Nenhum arquivo de referência fornecido. Por favor, forneça pelo menos um arquivo MIDI, áudio ou link do YouTube.")
        print("Exemplo de uso: python main.py --references data/reference_midi/*.mid  (ou só `python main.py` para o formulário)")
        sys.exit(1)

    # Elimina duplicatas preservando ordem
    ref_files = list(dict.fromkeys(ref_files))
    ref_audio += [audio for audio in map(_reference_audio_for, ref_files) if audio]

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

    style = STYLES[args.style]
    groove = _build_groove(ref_audio, style.fallback_groove)
    groove_source = "extraído do áudio de referência" if ref_audio else f"padrão do estilo {style.name} (sem áudio de referência)"

    key = detect_key(profile.pitch_class_distribution)
    bpm = args.bpm or (groove.tempo_bpm if ref_audio else profile.tempo_bpm)
    mode_name = "maior" if key.mode == "major" else "menor"
    print(f"  • Tonalidade estimada: {PITCH_CLASS_NAMES[key.tonic_pc]} {mode_name} | Andamento: {bpm:.0f} BPM")
    print(f"  • Estilo: {style.name} | Groove {groove_source}:")
    for part, pattern in groove.as_text(GROOVE_HIT_THRESHOLD).items():
        print(f"      {part:<5} {pattern}")

    sample_kit = None
    if args.timbre == "reference":
        if ref_audio:
            print("  • Recortando sons da referência para usar como instrumentos...")
            sample_kit = _build_sample_kit(ref_audio, args.output_dir)
        else:
            print("  • [Aviso] timbre 'reference' precisa de áudio de referência (YouTube/--audio); usando SoundFont.")

    # Configuração da função de fitness multiobjetivo
    evaluator = CorpusFitnessEvaluator(
        profile=profile,
        w_pitch_class=args.w_pitch,
        w_interval=args.w_interval,
        w_motif=args.w_motif,
        w_resolution=args.w_resolution,
        w_register=args.w_register,
        w_rhythm=args.w_rhythm
    )

    print("\n⚖️  Pesos do Fitness (normalizados: só a proporção entre eles importa):")
    for name, weight in evaluator.weights.items():
        print(f"  • {FITNESS_CRITERIA_LABELS[name]:<32} {weight:4.0%}")

    print(f"\n🧬 [2/4] Executando Algoritmo Genético para gerar {args.num_outputs} variações musicais...")
    random.seed(args.seed)

    # Para cada música anexada
    for song_idx in range(1, args.num_outputs + 1):
        random.seed(args.seed + (song_idx * 17))
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

        # 1. Salva o MIDI com arranjo harmonizado a partir da melodia evoluída
        band_pm = MusicArranger.create_full_band_midi(
            melody_individual=best_ind,
            pitch_class_distribution=profile.pitch_class_distribution,
            style=style,
            groove=groove,
            bpm=bpm,
            include_drums=not args.no_drums,
            include_bass=not args.no_bass,
            include_harmony=not args.no_harmony,
            key=key,
            humanize_seed=current_seed,
        )
        midi_filename = os.path.join(args.output_dir, f"penna_generated_{song_idx}.mid")
        band_pm.write(midi_filename)
        track_names = " + ".join(inst.name for inst in band_pm.instruments)
        print(f"  ✓ MIDI ({track_names}): {midi_filename} (Fitness: {best_ind.fitness:.4f})")
        _print_fitness_breakdown(evaluator, best_ind)

        # 2. Sintetiza em MP3 e WAV com ambiência de sala e saturação analógica
        try:
            convert_midi_to_audio(midi_filename, output_dir=args.output_dir, output_format="mp3", soundfont=args.soundfont, sample_kit=sample_kit)
        except Exception as e:
            print(f"  [Aviso] Falha na síntese de áudio: {e}")

        # 3. Salva gráfico de convergência do fitness
        plot_filename = os.path.join(args.output_dir, f"fitness_evolution_{song_idx}.png")
        engine.plot_fitness_history(plot_filename)

    print(f"\n✨ [3/4] Concluído com sucesso! Arquivos MIDI, MP3, WAV e gráficos salvos em '{args.output_dir}'.")
    print(f"🎧 [4/4] Áudios sintetizados e prontos para reprodução.")


if __name__ == "__main__":
    main()
