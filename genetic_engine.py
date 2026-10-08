"""
Genetic Algorithm Engine:
Executa a evolução populacional (seleção, crossover, mutação e registro de fitness).
"""

import copy
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from genetic_model import Gene, Individual
from fitness_evaluator import CorpusFitnessEvaluator


class GeneticEngine:
    """Motor de evolução para geração melódica algorítmica."""

    def __init__(
        self,
        evaluator: CorpusFitnessEvaluator,
        population_size: int = 60,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.25,
        elitism_count: int = 3,
        total_steps: int = 64,  # 4 compassos de 16 steps
        random_seed: int = 42
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.total_steps = total_steps
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.history_best_fitness: List[float] = []
        self.history_avg_fitness: List[float] = []

    def initialize_population(self) -> List[Individual]:
        min_p = max(36, self.evaluator.profile.min_pitch)
        max_p = min(96, self.evaluator.profile.max_pitch)
        allowed = list(range(min_p, max_p + 1))
        
        population = [
            Individual.create_random(total_steps=self.total_steps, allowed_pitches=allowed)
            for _ in range(self.population_size)
        ]
        return population

    def select_parent_tournament(self, population: List[Individual], k: int = 3) -> Individual:
        candidates = random.sample(population, k)
        candidates.sort(key=lambda ind: ind.fitness, reverse=True)
        return candidates[0]

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Crossover alinhado temporalmente
        # Encontra o ponto de corte acumulando durações
        child1_genes, child2_genes = [], []
        t1, t2 = 0, 0
        g1_idx, g2_idx = 0, 0
        cut_point = self.total_steps // 2  # metade da frase

        # Primeira metade
        for g in parent1.genes:
            if t1 + g.duration_steps <= cut_point:
                child1_genes.append(copy.deepcopy(g))
                t1 += g.duration_steps
            else:
                break

        for g in parent2.genes:
            if t2 + g.duration_steps <= cut_point:
                child2_genes.append(copy.deepcopy(g))
                t2 += g.duration_steps
            else:
                break

        # Preenche restante até total_steps trocando os genomas
        remaining_1 = self.total_steps - t1
        cur_rem = 0
        for g in parent2.genes:
            if cur_rem + g.duration_steps <= remaining_1:
                child1_genes.append(copy.deepcopy(g))
                cur_rem += g.duration_steps

        remaining_2 = self.total_steps - t2
        cur_rem2 = 0
        for g in parent1.genes:
            if cur_rem2 + g.duration_steps <= remaining_2:
                child2_genes.append(copy.deepcopy(g))
                cur_rem2 += g.duration_steps

        # Ajuste fino se faltou steps para completar a barra
        for child, rem in [(child1_genes, self.total_steps - sum(g.duration_steps for g in child1_genes)),
                           (child2_genes, self.total_steps - sum(g.duration_steps for g in child2_genes))]:
            if rem > 0:
                child.append(Gene(pitch=int(self.evaluator.profile.avg_pitch), duration_steps=rem, is_rest=False))

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def mutate(self, individual: Individual) -> Individual:
        min_p = self.evaluator.profile.min_pitch
        max_p = self.evaluator.profile.max_pitch

        for gene in individual.genes:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(["pitch_shift", "invert_rest", "octave_jump"])
                if mutation_type == "pitch_shift":
                    # Pequeno semitom / tom para manter coerência
                    shift = random.choice([-2, -1, 1, 2])
                    new_pitch = int(np.clip(gene.pitch + shift, min_p, max_p))
                    gene.pitch = new_pitch
                elif mutation_type == "invert_rest":
                    gene.is_rest = not gene.is_rest
                elif mutation_type == "octave_jump":
                    shift = random.choice([-12, 12])
                    new_pitch = int(np.clip(gene.pitch + shift, min_p, max_p))
                    gene.pitch = new_pitch
        return individual

    def run(self) -> Tuple[Individual, List[float], List[float]]:
        population = self.initialize_population()

        for gen in range(self.generations):
            # 1. Avalia população
            for ind in population:
                ind.fitness = self.evaluator.evaluate(ind)

            # Ordena por melhor fitness decrescente
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            best_fitness = population[0].fitness
            avg_fitness = float(np.mean([ind.fitness for ind in population]))
            self.history_best_fitness.append(best_fitness)
            self.history_avg_fitness.append(avg_fitness)

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"[Gen {gen:03d}/{self.generations}] Best Fitness: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

            # 2. Elitismo
            new_population = [copy.deepcopy(population[i]) for i in range(self.elitism_count)]

            # 3. Cruzamento e Mutação para compor a nova geração
            while len(new_population) < self.population_size:
                p1 = self.select_parent_tournament(population)
                p2 = self.select_parent_tournament(population)
                c1, c2 = self.crossover(p1, p2)
                new_population.append(self.mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(c2))

            population = new_population

        # Avaliação final
        for ind in population:
            ind.fitness = self.evaluator.evaluate(ind)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        return population[0], self.history_best_fitness, self.history_avg_fitness

    def plot_fitness_history(self, output_path: str = "fitness_history.png") -> None:
        """Gera e salva o gráfico obrigatório de evolução do fitness."""
        plt.figure(figsize=(9, 5))
        plt.plot(self.history_best_fitness, label="Melhor Fitness", color="#2ca02c", linewidth=2)
        plt.plot(self.history_avg_fitness, label="Fitness Médio", color="#1f77b4", linestyle="--")
        plt.title("Evolução do Fitness ao Longo das Gerações (Algoritmo Genético)")
        plt.xlabel("Geração")
        plt.ylabel("Fitness Normalizado")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Info] Gráfico de evolução de fitness salvo em: {output_path}")
