import random
import csv
import os
import time
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class KnapsackProblem:
    def __init__(self, capacity: int, items: List[Tuple[int, int]]):
        self.capacity = capacity
        self.items = items

class GeneticAlgorithm:
    def __init__(self, problem: KnapsackProblem, pop_size: int, crossover_method: str,
                 mutation_rate: float, init_method: str, stop_criterion: str, 
                 elite_size: int = 2):
        self.problem = problem
        self.pop_size = pop_size
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.init_method = init_method
        self.stop_criterion = stop_criterion
        self.elite_size = elite_size

    def initialize_population(self) -> List[List[bool]]:
        """Inicializa a populacao usando metodo aleatorio ou heuristico"""
        if self.init_method == 'random':
            return [self._create_random_solution() for _ in range(self.pop_size)]
        elif self.init_method == 'heuristic':
            return self._create_heuristic_population()
        else:
            raise ValueError("Método de inicialização desconhecido")

    def _create_random_solution(self) -> List[bool]:
        """Cria solucao com 15% de chance de selecionar cada item"""
        return [random.random() < 0.15 for _ in self.problem.items]

    def _create_heuristic_population(self) -> List[List[bool]]:
        """Cria populacao baseada na razao valor/peso"""
        sorted_items = sorted(enumerate(self.problem.items), 
                            key=lambda x: x[1][1]/x[1][0], reverse=True)
        selected = [False] * len(self.problem.items)
        total_weight = 0
        
        for idx, (w, v) in sorted_items:
            if total_weight + w <= self.problem.capacity:
                selected[idx] = True
                total_weight += w
        
        population = [selected.copy() for _ in range(self.pop_size//4)]
        population += [self._create_random_solution() for _ in range(self.pop_size - len(population))]
        return population

    def fitness(self, solution: List[bool]) -> float:
        """Calcula fitness com penalidade para solucoes invalidas"""
        total_value = 0
        total_weight = 0
        for i, selected in enumerate(solution):
            if selected:
                total_value += self.problem.items[i][1]
                total_weight += self.problem.items[i][0]
        
        if total_weight > self.problem.capacity:
            return total_value * 0.1
        return total_value

    def select_parents(self, population: List[List[bool]], fitnesses: List[float]) -> List[List[bool]]:
        """Selecao por roleta viciada"""
        total_fitness = sum(f for f in fitnesses if f > 0)
        probabilities = [f/total_fitness if f > 0 else 0 for f in fitnesses]
        return random.choices(population, weights=probabilities, k=2)

    def crossover(self, parent1: List[bool], parent2: List[bool]) -> List[List[bool]]:
        """Implementa tres metodos de crossover"""
        if self.crossover_method == 'one_point':
            point = random.randint(1, len(parent1)-1)
            return [
                parent1[:point] + parent2[point:],
                parent2[:point] + parent1[point:]
            ]
        elif self.crossover_method == 'two_point':
            points = sorted(random.sample(range(1, len(parent1)), 2))
            return [
                parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:],
                parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
            ]
        elif self.crossover_method == 'uniform':
            return [
                [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)],
                [p2 if random.random() < 0.5 else p1 for p1, p2 in zip(parent1, parent2)]
            ]
        else:
            raise ValueError(f"Método de crossover inválido: {self.crossover_method}")

    def mutate(self, solution: List[bool]) -> List[bool]:
        """Aplica mutacao bit a bit"""
        return [not gene if random.random() < self.mutation_rate else gene for gene in solution]

    def run(self) -> Tuple[List[bool], float, List[float]]:
        """Executa o algoritmo genetico"""
        population = self.initialize_population()
        best_solution = None
        best_fitness = -float('inf')
        history = []
        generations, no_improvement = 0, 0

        while True:
            
            fitnesses = [self.fitness(ind) for ind in population]
            
            
            current_best = max(fitnesses)
            if current_best > best_fitness:
                best_fitness = current_best
                best_solution = population[fitnesses.index(current_best)]
                no_improvement = 0
            else:
                no_improvement += 1
            
            history.append(current_best)

            
            if self.stop_criterion.startswith('generations'):
                max_gen = int(self.stop_criterion.split('_')[1])
                if generations >= max_gen:
                    break
            elif self.stop_criterion.startswith('convergence'):
                conv_limit = int(self.stop_criterion.split('_')[1])
                if no_improvement >= conv_limit:
                    break

            
            elite = sorted(zip(population, fitnesses), key=lambda x: -x[1])[:self.elite_size]
            new_population = [ind for ind, _ in elite]
            
            
            while len(new_population) < self.pop_size:
                parents = self.select_parents(population, fitnesses)
                offspring = self.crossover(parents[0], parents[1])
                for child in offspring:
                    new_population.append(self.mutate(child))
            
            population = new_population[:self.pop_size]
            generations += 1

        return best_solution, best_fitness, history

    def decode_solution(self, solution: List[bool]) -> Dict:
        selected_items = [i+1 for i, gene in enumerate(solution) if gene]
        total_weight = sum(self.problem.items[i][0] for i, gene in enumerate(solution) if gene)
        total_value = sum(self.problem.items[i][1] for i, gene in enumerate(solution) if gene)
        return {
            'selected_items': selected_items,
            'total_weight': total_weight,
            'total_value': total_value,
            'is_valid': total_weight <= self.problem.capacity
        }

def load_problem(file_path: str) -> KnapsackProblem:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        
        items = []
        capacity = 0
        
        for row in reader:
            if not row:
                continue
            if 'Capacidade' in row[0]:
                capacity = int(row[1])
                break
            if len(row) >= 3:
                items.append((int(row[1]), int(row[2])))
        
        return KnapsackProblem(capacity, items)

def run_experiments(problem_files: List[str]):
    """Executa experimentos com diferentes configuracoes"""
    configs = {
        'crossover': ['one_point', 'two_point', 'uniform'],
        'mutation_rate': [0.01, 0.05, 0.1],
        'init_method': ['random', 'heuristic'],
        'stop_criterion': ['generations_200', 'convergence_30']
    }

    results = []
    plot_dir = Path('results')
    plot_dir.mkdir(exist_ok=True)

    for file in problem_files:
        problem = load_problem(file)
        print(f"\n{'='*40}\nTestando: {Path(file).name}")
        print(f"Capacidade: {problem.capacity} | Itens: {len(problem.items)}\n{'='*40}")

        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.flatten()
        plt.suptitle(f"Evolução - {Path(file).name}", y=1.02)

        config_count = 0
        for crossover in configs['crossover']:
            for mutation in configs['mutation_rate']:
                for init_method in configs['init_method']:
                    for stop in configs['stop_criterion']:
                        ga = GeneticAlgorithm(
                            problem=problem,
                            pop_size=100,
                            crossover_method=crossover,
                            mutation_rate=mutation,
                            init_method=init_method,
                            stop_criterion=stop
                        )

                        start = time.time()
                        solution, fitness, history = ga.run()
                        exec_time = time.time() - start
                        decoded = ga.decode_solution(solution)

                        results.append({
                            'Arquivo': Path(file).name,
                            'Crossover': crossover,
                            'Mutação': mutation,
                            'Inicialização': init_method,
                            'Critério Parada': stop,
                            'Valor': decoded['total_value'],
                            'Peso': decoded['total_weight'],
                            'Válido': decoded['is_valid'],
                            'Tempo (s)': round(exec_time, 2),
                            'Gerações': len(history)
                        })

                        if config_count < 6:
                            axs[config_count].plot(history)
                            axs[config_count].set_title(
                                f"{crossover} | {mutation} | {init_method}\n"
                                f"Valor: {decoded['total_value']} Peso: {decoded['total_weight']}"
                            )
                            axs[config_count].grid(True)
                            config_count += 1

                        print(f"\nConfig: {crossover}, {mutation}, {init_method}, {stop}")
                        print(f"Itens: {decoded['selected_items']}")
                        print(f"Valor: {decoded['total_value']} | Peso: {decoded['total_weight']}/{problem.capacity}")
                        print(f"Tempo: {exec_time:.2f}s | Gerações: {len(history)}")

        plt.tight_layout()
        plt.savefig(plot_dir / f"{Path(file).stem}.png", bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv(plot_dir / 'resultados_completos.csv', index=False)
    
    print("\n\nRESULTADOS FINAIS:")
    print(df[['Arquivo', 'Crossover', 'Mutação', 'Inicialização', 'Valor', 'Peso', 'Válido']]
          .to_markdown(index=False, tablefmt="grid"))
    
    print(f"\nDados completos salvos em: {(plot_dir / 'resultados_completos.csv').absolute()}")

if __name__ == "__main__":
    import re
    csv_files = sorted(
        [f for f in os.listdir() if f.startswith('knapsack_') and f.endswith('.csv')],
        key=lambda x: int(re.search(r'knapsack_(\d+)\.csv', x).group(1))
    )
    
    if not csv_files:
        print("Nenhum arquivo CSV encontrado na pasta!")
    else:
        run_experiments(csv_files)