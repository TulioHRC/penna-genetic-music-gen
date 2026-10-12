import copy
import random
from dataclasses import replace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from alg_genetic_model import Gene, Individual
from alg_genetic_fitness_evaluator import (
    MAX_INTERVAL_SEMITONES,
    CorpusFitnessEvaluator,
)


# Separa os genes em duas listas no passo especificado. Se um gene atravessar o ponto de corte, ele será dividido em dois. O cruzamento ocorre no meio da peça, recombinando partes dos pais para gerar os filhos.
def _dividir_no_passo(
    genes: List[Gene],
    passo_corte: int,
) -> Tuple[List[Gene], List[Gene]]:
    inicio = []
    final = []
    passo_atual = 0

    for gene in genes:
        fim_gene = passo_atual + gene.duration_steps

        if fim_gene <= passo_corte:
            inicio.append(copy.deepcopy(gene))

        elif passo_atual >= passo_corte:
            final.append(copy.deepcopy(gene))

        else:
            # Divide o gene que atravessa o ponto de corte.
            inicio.append(
                replace(
                    gene,
                    duration_steps=passo_corte - passo_atual,
                )
            )

            final.append(
                replace(
                    gene,
                    duration_steps=fim_gene - passo_corte,
                )
            )

        passo_atual = fim_gene

    return inicio, final


class GeneticEngine:

    def __init__(
        self,
        avaliador: CorpusFitnessEvaluator,
        tamanho_populacao: int = 60,
        geracoes: int = 100,
        taxa_cruzamento: float = 0.8,
        taxa_mutacao: float = 0.25,
        quantidade_elitismo: int = 3,
        # Quantidade de indivíduos preservados sem alterações.
        total_passos: int = 64,
        # Quatro compassos de 16 passos.
        semente_aleatoria: int = 42,
    ):
        self.avaliador = avaliador
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.quantidade_elitismo = quantidade_elitismo
        self.total_passos = total_passos

        # Define sementes para tornar os resultados reproduzíveis.
        random.seed(semente_aleatoria)
        np.random.seed(semente_aleatoria)

        # Obtém a distribuição de intervalos musicais da referência. Descarta intervalos fora do limite permitido.
        # Se não houver intervalos disponíveis   utiliza zero semitons.
        intervalos = {
            intervalo: probabilidade
            for intervalo, probabilidade in (
                avaliador.profile.interval_transition_distribution.items()
            )
            if abs(intervalo) <= MAX_INTERVAL_SEMITONES
        } or {0: 1.0}

        self._intervalos = list(intervalos)
        self._probabilidades_intervalos = (
            np.array(list(intervalos.values()))
            / sum(intervalos.values())
        )

        # Armazena o histórico da evolução da aptidão.
        self.historico_melhor_aptidao: List[float] = []
        self.historico_aptidao_media: List[float] = []

    def _sortear_intervalo(self) -> int:
        return int(
            np.random.choice(
                self._intervalos,
                p=self._probabilidades_intervalos,
            )
        )

    def _ajustar_a_tessitura(self, altura_nota: int) -> int:
        altura_minima = self.avaliador.profile.min_pitch
        altura_maxima = self.avaliador.profile.max_pitch

        # Ajusta a nota por oitavas, quando possível.
        while (
            altura_nota > altura_maxima
            and altura_nota - 12 >= altura_minima
        ):
            altura_nota -= 12

        while (
            altura_nota < altura_minima
            and altura_nota + 12 <= altura_maxima
        ):
            altura_nota += 12

        # Garante que a nota permaneça dentro dos limites.
        return int(
            np.clip(
                altura_nota,
                altura_minima,
                altura_maxima,
            )
        )

    def _gerar_melodia_aleatoria(
        self,
        individuo: Individual,
    ) -> Individual:
        altura_central = int(self.avaliador.profile.avg_pitch)

        # Escolhe uma nota inicial próxima da altura média.
        altura_nota = self._ajustar_a_tessitura(
            altura_central + random.randint(-6, 6) # aleatório dentro de uma sexta menor acima ou abaixo da média
        )

        for gene in individuo.genes:
            if not gene.is_rest:
                gene.pitch = altura_nota

                # Sorteia a próxima nota conforme os intervalos
                # encontrados na música de referência.
                altura_nota = self._ajustar_a_tessitura(
                    altura_nota + self._sortear_intervalo()
                )

        return individuo

    def inicializar_populacao(self) -> List[Individual]:
        return [
            self._gerar_melodia_aleatoria(
                Individual.create_random(
                    total_steps=self.total_passos
                )
            )
            for _ in range(self.tamanho_populacao)
        ]

    def selecionar_pai_por_torneio(
        self,
        populacao: List[Individual],
        quantidade_candidatos: int = 3,
    ) -> Individual:
        candidatos = random.sample(
            populacao,
            quantidade_candidatos,
        )

        candidatos.sort(
            key=lambda individuo: individuo.fitness,
            reverse=True,
        )

        return candidatos[0]

    def cruzar_individuos(
        self,
        pai1: Individual,
        pai2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        Realiza o cruzamento entre dois indivíduos.

        Filho 1: início do pai 1 + final do pai 2.
        Filho 2: início do pai 2 + final do pai 1.
        """
        if random.random() > self.taxa_cruzamento:
            return (
                copy.deepcopy(pai1),
                copy.deepcopy(pai2),
            )

        # Define o ponto de corte no meio da peça.
        passo_corte = self.total_passos // 2

        inicio_pai1, final_pai1 = _dividir_no_passo(
            pai1.genes,
            passo_corte,
        )

        inicio_pai2, final_pai2 = _dividir_no_passo(
            pai2.genes,
            passo_corte,
        )

        filho1 = Individual(
            genes=inicio_pai1 + final_pai2
        )

        filho2 = Individual(
            genes=inicio_pai2 + final_pai1
        )

        return filho1, filho2

    def mutar_individuo(
        self,
        individuo: Individual,
    ) -> Individual:
        altura_minima = self.avaliador.profile.min_pitch
        altura_maxima = self.avaliador.profile.max_pitch

        altura_anterior = None

        for gene in individuo.genes:

            # Decide se o gene sofrerá mutação.
            if random.random() < self.taxa_mutacao:

                # Escolhe o tipo de mutação.
                tipo_mutacao = random.choice([
                    "deslocar_altura",
                    "inverter_pausa",
                    "salto_oitava",
                    "sortear_novo_intervalo",
                ])

                if tipo_mutacao == "deslocar_altura":
                    # Desloca a nota em um ou dois semitons.
                    deslocamento = random.choice([-2, -1, 1, 2])

                    nova_altura = int(
                        np.clip(
                            gene.pitch + deslocamento,
                            altura_minima,
                            altura_maxima,
                        )
                    )

                    gene.pitch = nova_altura

                elif tipo_mutacao == "inverter_pausa":
                    gene.is_rest = not gene.is_rest

                elif tipo_mutacao == "salto_oitava":
                    # Transpõe a nota uma oitava para cima ou para baixo.
                    deslocamento = random.choice([-12, 12])

                    nova_altura = int(
                        np.clip(
                            gene.pitch + deslocamento,
                            altura_minima,
                            altura_maxima,
                        )
                    )

                    gene.pitch = nova_altura

                elif (
                    tipo_mutacao == "sortear_novo_intervalo"
                    and altura_anterior is not None
                ):
                    # Sorteia um novo intervalo a partir da nota anterior.
                    gene.pitch = self._ajustar_a_tessitura(
                        altura_anterior + self._sortear_intervalo()
                    )

            # Atualiza a altura da última nota válida.
            if not gene.is_rest:
                altura_anterior = gene.pitch

        return individuo

    def run(
        self,
    ) -> Tuple[Individual, List[float], List[float]]:
        populacao = self.inicializar_populacao()

        for geracao in range(self.geracoes):

            # 1. Avalia a aptidão de cada indivíduo.
            for individuo in populacao:
                individuo.fitness = self.avaliador.evaluate(individuo)

            # Ordena a população da maior para a menor aptidão.
            populacao.sort(
                key=lambda individuo: individuo.fitness,
                reverse=True,
            )

            # Registra as estatísticas da geração.
            melhor_aptidao = populacao[0].fitness

            aptidao_media = float(
                np.mean([
                    individuo.fitness
                    for individuo in populacao
                ])
            )

            self.historico_melhor_aptidao.append(
                melhor_aptidao
            )

            self.historico_aptidao_media.append(
                aptidao_media
            )

            # Exibe o progresso periodicamente.
            if geracao % 10 == 9 or geracao == 0:
                print(
                    f"[Geração {geracao + 1:03d}/{self.geracoes}] "
                    f"Melhor aptidão: {melhor_aptidao:.4f} | "
                    f"Média: {aptidao_media:.4f}"
                )

            # 2. Elitismo:
            # preserva os melhores indivíduos sem alterações.
            nova_populacao = [
                copy.deepcopy(populacao[i])
                for i in range(self.quantidade_elitismo)
            ]

            # 3. Seleção, cruzamento e mutação.
            # Continua até completar a nova população.
            while len(nova_populacao) < self.tamanho_populacao:

                pai1 = self.selecionar_pai_por_torneio(populacao)
                pai2 = self.selecionar_pai_por_torneio(populacao)

                filho1, filho2 = self.cruzar_individuos(
                    pai1,
                    pai2,
                )

                nova_populacao.append(
                    self.mutar_individuo(filho1)
                )

                # Adiciona o segundo filho se ainda houver espaço.
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(
                        self.mutar_individuo(filho2)
                    )

            populacao = nova_populacao

        # 4. Avaliação final da população.
        for individuo in populacao:
            individuo.fitness = self.avaliador.evaluate(individuo)

        populacao.sort(
            key=lambda individuo: individuo.fitness,
            reverse=True,
        )

        return (
            populacao[0],
            self.historico_melhor_aptidao,
            self.historico_aptidao_media,
        )

    def plotar_historico_aptidao(
        self,
        caminho_saida: str = "historico_aptidao.png",
    ) -> None:
        """Gera e salva o gráfico da evolução da aptidão."""

        plt.figure(figsize=(9, 5))

        plt.plot(
            self.historico_melhor_aptidao,
            label="Melhor aptidão",
            color="#2ca02c",
            linewidth=2,
        )

        plt.plot(
            self.historico_aptidao_media,
            label="Aptidão média",
            color="#1f77b4",
            linestyle="--",
        )

        plt.title(
            "Evolução da aptidão ao longo das gerações"
        )

        plt.xlabel("Geração")
        plt.ylabel("Aptidão normalizada")

        plt.grid(
            True,
            linestyle=":",
            alpha=0.6,
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            caminho_saida,
            dpi=300,
        )

        plt.close()

        print(
            f"[Informação] Gráfico de evolução da aptidão "
            f"salvo em: {caminho_saida}"
        )