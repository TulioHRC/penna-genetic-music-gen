# penna-genetic-music-gen 🎸🧬

Sistema de Composição Musical Algorítmica utilizando **Algoritmos Genéticos** no domínio simbólico (MIDI), desenvolvido no contexto da disciplina DCC831 – IA Generativa para Música (DCC/UFMG).

---

## 🎯 Objetivo & Restrição Estilística

- **Objetivo**: Geração de peças musicais em domínio simbólico (`.mid`) a partir de restrições estruturais e estilísticas evoluídas via computação evolucionária.
- **Inspiração / Restrição Estilística**: Estilo musical e sonoridade melódica inspirados no Rock Nacional/Pop Rock (referência: Pitty / "Penna").
- **Saídas**: Arquivos MIDI (`.mid`) reprodutíveis e arquivos sintetizados em áudio (`.wav`/`.mp3`) para audição e avaliação qualitativa.

---

## 🧬 Especificação Técnica do Algoritmo Genético

O sistema implementa o ciclo clássico de Algoritmos Genéticos voltado à música simbólica:

1. **Representação do Genótipo**:
   - Codificação de eventos musicais: sequências de notas/pausas (pitch, duração rítmica, onset/tempo de início, dinâmica/velocity).
   - Estrutura em compassos (ex.: 4/4) ou blocos harmônico-melódicos.
2. **População Inicial**:
   - Geração pseudo-aleatória ou semi-estruturada (respeitando escalas/tonalidades predefinidas como semente).
3. **Função de Fitness (Avaliação)**:
   - Heurísticas baseadas em Teoria Musical e contorno melódico:
     - Consonância e adesão à escala/tonalidade (ex.: menor harmônica / pentatônica / diatônica).
     - Penalização de saltos intervalares excessivos e dissonâncias descontextualizadas.
     - Variedade rítmica balanceada (evitar notas monótonas ou ritmos caóticos).
     - Densidade de notas e coerência estrutural (fraseado musical).
4. **Operadores Genéticos**:
   - **Seleção**: Torneio ou Roleta (com elitismo para preservar os melhores indivíduos).
   - **Cruzamento (Crossover)**: Crossover de ponto único, dois pontos ou por compassos/frases.
   - **Mutação**: Alteração de tom (pitch shift / transposição), divisão/fusão de notas no tempo, inversão ou perturbação rítmica.
5. **Critério de Parada**:
   - Convergência de fitness médio/máximo ou limite estipulado de gerações.
   - Geração obrigatória do gráfico de convergência do fitness ao longo das gerações.

---

## 📄 Requisitos do Resumo Científico (ISMIR)

Submissão de artigo curto no padrão de conferência ISMIR (template oficial `ismir.net`), contendo:
- Limite de até 2 páginas de conteúdo (+ referências ilimitadas).
- Estrutura:
  1. **Introdução**: Motivação, formulação do problema e escolha do método evolucionário.
  2. **Trabalhos Relacionados**: Contextualização com literatura de algoritmos genéticos aplicados à composição algorítmica.
  3. **Método**: Detalhamento formal (genótipo, fenótipo, funções heurísticas de fitness, operadores genéticos, hiperparâmetros).
  4. **Resultados**: Análise quantitativa (gráfico de evolução de fitness) e qualitativa das composições geradas sob diferentes sementes/hiperparâmetros.
  5. **Discussão e Trabalhos Futuros**: Limitações, potencial de hibridização e próximas extensões.
  6. **Referências Bibliográficas**.

---

## 📦 Entregáveis do Projeto (Moodle & GitHub)

1. **Repositório GitHub**:
   - Código-fonte modular e documentado.
   - `requirements.txt` com dependências (ex.: `music21`, `pretty_midi`, `mido`, `matplotlib`, `numpy`).
   - Instruções de execução e reprodutibilidade de experimentos.
2. **Áudios & Mídias**:
   - Pelo menos **3 peças musicais geradas** sintetizadas em áudio a partir dos MIDIs gerados, explorando variações de sementes e hiperparâmetros (taxa de mutação, pesos de fitness, etc.).
3. **Resumo ISMIR em PDF**:
   - Documento final compilado com o gráfico de fitness e links para o áudio.
4. **Declaração sobre o Uso de IA**:
   - Seção explícita no README descrevendo o apoio de ferramentas de IA (geração auxiliar, depuração, revisão) com registro de escopo e links das conversas, conforme a diretriz da disciplina.

## 🚀 Como Executar

### 1. Instalação de dependências:
```bash
pip install --no-deps tinysoundfont   # antes do requirements: evita compilar pyaudio
pip install -r requirements.txt
```

Timbre do áudio gerado (`"timbre"` no `config.json`):
- `reference` (padrão): recorta bumbo, caixa, chimbal, uma nota de baixo e uma de lead da própria música de referência (`sample_kit.py`) e toca o arranjo com eles. Os sons extraídos ficam em `output/sample_kit/` para conferência. Precisa de áudio de referência (YouTube ou `--audio`).
- `soundfont`: instrumentos General MIDI da SoundFont abaixo (também completa o que o modo `reference` não extrai, como pratos e acordes).

SoundFont com samples de instrumentos (sem ela o áudio sai com o sintetizador aditivo, bem mais artificial):
```bash
mkdir -p data/soundfonts
curl -L -o data/soundfonts/GeneralUser-GS.sf2 \
  https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2
```

### 2. Modos de Uso da CLI:

#### A) Passando links do YouTube diretamente (baixa áudio, transcreve para MIDI e roda o GA):
```bash
python main.py --youtube "https://www.youtube.com/watch?v=EXEMPLO_PITTY"
```

#### B) Passando arquivos locais de áudio (.mp3, .wav, .mp4):
```bash
python main.py --audio "minhas_musicas/pitty_mascara.mp3"
```

#### C) Passando arquivos .mid de referência:
```bash
python main.py --references data/reference_midi/*.mid --generations 80 --population 60
```

#### D) Customizando os pesos da Função de Fitness Multiobjetivo:
```bash
python main.py --references data/reference_midi/*.mid \
  --w-pitch 0.25 \
  --w-interval 0.25 \
  --w-motif 0.20 \
  --w-resolution 0.10 \
  --w-register 0.10 \
  --w-rhythm 0.10
```

### 3. Saídas Geradas:
Tudo é salvo na pasta `output/`:
- `penna_generated_1.mid`, `penna_generated_2.mid`, `penna_generated_3.mid` (Músicas simbólicas MIDI)
- `penna_generated_1.musicxml` (Partitura estruturada MusicXML)
- `penna_generated_1.mp3`, `penna_generated_1.wav` (Áudios sintetizados com bateria acústica, distorção valvulada e ambiência)
- `fitness_evolution_1.png`, `fitness_evolution_2.png`, `fitness_evolution_3.png` (Gráficos de convergência do fitness)

---

## 🤖 Declaração de Uso de IA Generativa

*(Em conformidade com as diretrizes do DCC831 / UFMG)*

**Ferramentas:** Antigravity (Gemini), na versão inicial do projeto, e Claude Code (Anthropic, modelo Claude Opus), nas iterações seguintes.

### Código auxiliar (fora da lógica do método)

A IA foi usada para **gerar o código auxiliar**: tudo o que cerca o algoritmo genético para
que ele receba referências reais e produza algo que se possa ouvir. Em cada caso eu descrevi o
comportamento desejado; a IA escreveu o código, e os problemas que apareceram ao ouvir os
resultados foram depurados com ela ao longo das conversas.

- **Interface e configuração** (`main.py`, `config.py`, `config.json`, `tui.py`): a CLI, a
  leitura dos valores padrão de um `config.json` e o formulário interativo no terminal
  (Textual), que monta os campos a partir das opções da CLI e salva novos padrões. A IA também
  acrescentou a tabela que mostra, ao fim de cada peça, a nota e a contribuição de cada critério
  do fitness, usada para ajustar os pesos.
- **Download e transcrição** (`audio_transcriber.py`): download de áudio do YouTube com
  `yt-dlp` e transcrição da linha melódica para MIDI com YIN. A IA também ajudou a depurar o
  download (bloqueio "not a bot", cookies, runtime JavaScript exigido pelo `yt-dlp`) e a
  conversão para WAV estéreo em 44,1 kHz.
- **Perfil do corpus** (`corpus_analyzer.py`): histograma de notas, distribuição de intervalos,
  tessitura e andamento das referências, que alimentam o fitness.
- **Extração de groove** (`groove_extractor.py`): estimativa do andamento e do padrão de
  bateria de um compasso (bumbo, caixa, chimbal) a partir do áudio de referência, por separação
  harmônico/percussiva e detecção de batidas.
- **Arranjo** (`music_arranger.py`, `styles.py`): transforma a melodia evoluída em um arranjo
  para audição, com detecção de tonalidade, um acorde por compasso, baixo travado no bumbo,
  bateria no groove extraído e humanização de tempo e intensidade.
- **Sintetizadores** (`sample_kit.py`, `sample_renderer.py`, `soundfont_renderer.py`,
  `audio_synth.py`): convertem o MIDI em áudio tocando o arranjo com sons recortados da própria
  música de referência ou com uma SoundFont. A depuração feita com a IA incluiu evitar a voz
  no sample de lead (canal lateral L−R) e remover cliques, vazamentos e aliasing dos samples.
- **Documentação e relatório** (`README.md`, `requirements.txt`, `report/`): a IA gerou a
  documentação, a estrutura do relatório ISMIR, a descrição técnica da seção Método (a partir
  do código) e a bibliografia. As seções autorais do relatório (introdução, trabalhos
  relacionados, resultados e discussão) são escritas pelo autor.

Nada disso altera a melodia avaliada pelo algoritmo genético: o arranjo e a síntese servem
apenas para a audição.