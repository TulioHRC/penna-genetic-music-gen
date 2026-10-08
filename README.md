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
pip install -r requirements.txt
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

#### D) Modo Demonstração (usa riffs sintéticos embutidos se nada for passado):
```bash
python main.py
```

### 3. Saídas Geradas:
Tudo é salvo na pasta `output/`:
- `penna_generated_1.mid`, `penna_generated_2.mid`, `penna_generated_3.mid` (Músicas simbólicas)
- `penna_generated_1.mp3`, `penna_generated_2.mp3`, `penna_generated_3.mp3` (Áudios gerados prontos para escuta)
- `fitness_evolution_1.png`, `fitness_evolution_2.png`, `fitness_evolution_3.png` (Gráficos de convergência do fitness)

---

## 🤖 Declaração de Uso de IA Generativa

*(Em conformidade com as diretrizes do DCC831 / UFMG)*:
- Este projeto contou com suporte de IA generativa (Antigravity / Gemini) como assistente para estruturação de documentação, scaffolding de código e planejamento dos operadores de fitness. A concepção dos experimentos, validação técnica e análise musical foram revisadas e conduzidas pelo autor.