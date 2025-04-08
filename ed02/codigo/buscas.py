import numpy as np
import time
import heapq
import csv
from collections import deque
import psutil
import os
import copy

class Puzzle:
    def __init__(self, estado, pai=None, acao=None, custo=0):
        self.estado = estado  # matriz 3x3 representando o tabuleiro
        self.pai = pai        # nó pai na árvore de busca
        self.acao = acao      # ação que levou a este estado
        self.custo = custo    # custo do caminho até aqui
        self.blank_pos = self.encontrar_vazio()
    
    def encontrar_vazio(self):
        """Encontra a posição do espaço vazio (0)"""
        for i in range(3):
            for j in range(3):
                if self.estado[i][j] == 0:
                    return (i, j)
        return None
    
    def gerar_sucessores(self):
        """Gera todos os estados possíveis a partir do estado atual"""
        sucessores = []
        
        # Movimentos possíveis: cima, baixo, esquerda, direita
        movimentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nomes_movimentos = ["cima", "baixo", "esquerda", "direita"]
        
        i, j = self.blank_pos
        
        for idx, (di, dj) in enumerate(movimentos):
            novo_i, novo_j = i + di, j + dj
            
            # Verifica se o novo estado é válido
            if 0 <= novo_i < 3 and 0 <= novo_j < 3:
                # Cria uma cópia do estado atual
                novo_estado = copy.deepcopy(self.estado)
                
                # Troca o espaço vazio com a peça adjacente
                novo_estado[i][j] = novo_estado[novo_i][novo_j]
                novo_estado[novo_i][novo_j] = 0
                
                # Cria um novo nó
                sucessor = Puzzle(novo_estado, self, nomes_movimentos[idx], self.custo + 1)
                
                sucessores.append(sucessor)
        
        return sucessores
    
    def eh_objetivo(self, estado_objetivo):
        """Verifica se o estado atual é o estado objetivo"""
        return np.array_equal(self.estado, estado_objetivo)
    
    def __lt__(self, outro):
        """Para comparação na fila de prioridade"""
        return self.custo < outro.custo
    
    def __eq__(self, outro):
        """Para verificar se dois estados são iguais"""
        return np.array_equal(self.estado, outro.estado)
    
    def __hash__(self):
        """Para usar como chave em conjuntos e dicionários"""
        return hash(str(self.estado))
    
    def __str__(self):
        """Representação em string do estado"""
        return '\n'.join([' '.join(map(str, row)) for row in self.estado])


# Algoritmos de Busca

def busca_largura(estado_inicial, estado_objetivo):
    """Busca em Largura (BFS)"""
    inicio = time.time()
    memoria_pico = 0
    
    nó_inicial = Puzzle(estado_inicial)
    
    if nó_inicial.eh_objetivo(estado_objetivo):
        fim = time.time()
        return nó_inicial, 1, fim - inicio, memoria_pico
    
    fronteira = deque([nó_inicial])
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        # Medir memória durante a execução
        memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memoria_pico = max(memoria_pico, memoria_atual)
        
        nó = fronteira.popleft()
        nos_expandidos += 1
        
        # Adiciona o estado atual ao conjunto de estados explorados
        explorados.add(hash(str(nó.estado)))
        
        # Gera os sucessores
        for sucessor in nó.gerar_sucessores():
            # Verifica se este estado já foi explorado
            if hash(str(sucessor.estado)) not in explorados:
                if sucessor.eh_objetivo(estado_objetivo):
                    fim = time.time()
                    memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    memoria_pico = max(memoria_pico, memoria_atual)
                    return sucessor, nos_expandidos, fim - inicio, memoria_pico
                
                fronteira.append(sucessor)
                explorados.add(hash(str(sucessor.estado)))
    
    # Se chegou aqui, não encontrou solução
    fim = time.time()
    memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memoria_pico = max(memoria_pico, memoria_atual)
    return None, nos_expandidos, fim - inicio, memoria_pico


def busca_profundidade(estado_inicial, estado_objetivo, limite=100):
    """Busca em Profundidade (DFS) com limite de profundidade"""
    inicio = time.time()
    memoria_pico = 0
    
    nó_inicial = Puzzle(estado_inicial)
    
    if nó_inicial.eh_objetivo(estado_objetivo):
        fim = time.time()
        return nó_inicial, 1, fim - inicio, memoria_pico
    
    pilha = [(nó_inicial, 0)]  # (nó, profundidade)
    explorados = set()
    nos_expandidos = 0
    
    while pilha:
        # Medir memória durante a execução
        memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memoria_pico = max(memoria_pico, memoria_atual)
        
        nó, profundidade = pilha.pop()
        nos_expandidos += 1
        
        # Adiciona o estado atual ao conjunto de estados explorados
        explorados.add(hash(str(nó.estado)))
        
        # Se não ultrapassou o limite de profundidade
        if profundidade < limite:
            # Gera os sucessores
            for sucessor in nó.gerar_sucessores():
                # Verifica se este estado já foi explorado
                if hash(str(sucessor.estado)) not in explorados:
                    if sucessor.eh_objetivo(estado_objetivo):
                        fim = time.time()
                        memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                        memoria_pico = max(memoria_pico, memoria_atual)
                        return sucessor, nos_expandidos, fim - inicio, memoria_pico
                    
                    pilha.append((sucessor, profundidade + 1))
    
    # Se chegou aqui, não encontrou solução
    fim = time.time()
    memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memoria_pico = max(memoria_pico, memoria_atual)
    return None, nos_expandidos, fim - inicio, memoria_pico


# Funções heurísticas

def hamming_distance(estado, estado_objetivo):
    """Heurística de Hamming: número de peças fora do lugar"""
    distancia = 0
    for i in range(3):
        for j in range(3):
            if estado[i][j] != 0 and estado[i][j] != estado_objetivo[i][j]:
                distancia += 1
    return distancia

def manhattan_distance(estado, estado_objetivo):
    """Heurística de Manhattan: soma das distâncias de cada peça até sua posição correta"""
    distancia = 0
    
    # Encontra a posição de cada número no estado objetivo
    posicoes_objetivo = {}
    for i in range(3):
        for j in range(3):
            if estado_objetivo[i][j] != 0:
                posicoes_objetivo[estado_objetivo[i][j]] = (i, j)
    
    # Calcula a distância de Manhattan
    for i in range(3):
        for j in range(3):
            if estado[i][j] != 0:  # Ignora o espaço vazio
                i_obj, j_obj = posicoes_objetivo[estado[i][j]]
                distancia += abs(i - i_obj) + abs(j - j_obj)
    
    return distancia


def busca_gulosa(estado_inicial, estado_objetivo, heuristica=manhattan_distance):
    """Busca Gulosa (Best-First)"""
    inicio = time.time()
    memoria_pico = 0
    
    nó_inicial = Puzzle(estado_inicial)
    
    if nó_inicial.eh_objetivo(estado_objetivo):
        fim = time.time()
        return nó_inicial, 1, fim - inicio, memoria_pico
    
    # Fila de prioridade: (valor_heuristica, contador, nó)
    contador = 0
    fronteira = [(heuristica(nó_inicial.estado, estado_objetivo), contador, nó_inicial)]
    heapq.heapify(fronteira)
    
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        # Medir memória durante a execução
        memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memoria_pico = max(memoria_pico, memoria_atual)
        
        _, _, nó = heapq.heappop(fronteira)
        nos_expandidos += 1
        
        if nó.eh_objetivo(estado_objetivo):
            fim = time.time()
            memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memoria_pico = max(memoria_pico, memoria_atual)
            return nó, nos_expandidos, fim - inicio, memoria_pico
        
        # Adiciona o estado atual ao conjunto de estados explorados
        explorados.add(hash(str(nó.estado)))
        
        # Gera os sucessores
        for sucessor in nó.gerar_sucessores():
            # Verifica se este estado já foi explorado
            if hash(str(sucessor.estado)) not in explorados:
                contador += 1
                heapq.heappush(fronteira, (heuristica(sucessor.estado, estado_objetivo), contador, sucessor))
    
    # Se chegou aqui, não encontrou solução
    fim = time.time()
    memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memoria_pico = max(memoria_pico, memoria_atual)
    return None, nos_expandidos, fim - inicio, memoria_pico


def busca_a_estrela(estado_inicial, estado_objetivo, heuristica=manhattan_distance):
    """Algoritmo A*"""
    inicio = time.time()
    memoria_pico = 0
    
    nó_inicial = Puzzle(estado_inicial)
    
    if nó_inicial.eh_objetivo(estado_objetivo):
        fim = time.time()
        return nó_inicial, 1, fim - inicio, memoria_pico
    
    # Fila de prioridade: (f(n) = g(n) + h(n), contador, nó)
    contador = 0
    fronteira = [(heuristica(nó_inicial.estado, estado_objetivo) + nó_inicial.custo, contador, nó_inicial)]
    heapq.heapify(fronteira)
    
    explorados = {}  # Dicionário de estados explorados -> custo
    nos_expandidos = 0
    
    while fronteira:
        # Medir memória durante a execução
        memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memoria_pico = max(memoria_pico, memoria_atual)
        
        _, _, nó = heapq.heappop(fronteira)
        nos_expandidos += 1
        
        if nó.eh_objetivo(estado_objetivo):
            fim = time.time()
            memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memoria_pico = max(memoria_pico, memoria_atual)
            return nó, nos_expandidos, fim - inicio, memoria_pico
        
        # Chave do estado atual
        estado_str = hash(str(nó.estado))
        
        # Adiciona o estado atual ao conjunto de estados explorados
        explorados[estado_str] = nó.custo
        
        # Gera os sucessores
        for sucessor in nó.gerar_sucessores():
            sucessor_str = hash(str(sucessor.estado))
            
            # Se o estado não foi explorado ou tem um custo menor
            if sucessor_str not in explorados or sucessor.custo < explorados[sucessor_str]:
                explorados[sucessor_str] = sucessor.custo
                contador += 1
                f_n = sucessor.custo + heuristica(sucessor.estado, estado_objetivo)
                heapq.heappush(fronteira, (f_n, contador, sucessor))
    
    # Se chegou aqui, não encontrou solução
    fim = time.time()
    memoria_atual = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memoria_pico = max(memoria_pico, memoria_atual)
    return None, nos_expandidos, fim - inicio, memoria_pico


def reconstruir_caminho(nó):
    """Reconstrói o caminho da solução"""
    caminho = []
    atual = nó
    
    while atual is not None:
        caminho.append(atual)
        atual = atual.pai
    
    return list(reversed(caminho))


def carregar_instancias(arquivo_csv):
    """Carrega instâncias do problema a partir de um arquivo CSV"""
    instancias = []
    
    with open(arquivo_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pula o cabeçalho
        
        for row in reader:
            # Converte as posições para inteiros e forma uma matriz 3x3
            estado = []
            for i in range(0, 9, 3):
                estado.append([int(row[i]), int(row[i+1]), int(row[i+2])])
            
            instancias.append(np.array(estado))
    
    return instancias


def comparar_algoritmos(instancias, estado_objetivo):
    """Compara os diferentes algoritmos de busca para as instâncias fornecidas"""
    
    resultados = []
    
    for i, instancia in enumerate(instancias):
        print(f"\nTestando instância {i+1}:")
        print(instancia)
        
        resultado_instancia = {"instancia": i+1}
        
        # BFS
        solucao, nos, tempo, memoria = busca_largura(instancia, estado_objetivo)
        if solucao:
            caminho = reconstruir_caminho(solucao)
            resultado_instancia["BFS"] = {
                "solucao_encontrada": True,
                "comprimento_caminho": len(caminho) - 1,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"BFS: Solução encontrada em {len(caminho) - 1} passos, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        else:
            resultado_instancia["BFS"] = {
                "solucao_encontrada": False,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"BFS: Solução não encontrada, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        
        # DFS (com limite de profundidade)
        solucao, nos, tempo, memoria = busca_profundidade(instancia, estado_objetivo, limite=30)
        if solucao:
            caminho = reconstruir_caminho(solucao)
            resultado_instancia["DFS"] = {
                "solucao_encontrada": True,
                "comprimento_caminho": len(caminho) - 1,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"DFS: Solução encontrada em {len(caminho) - 1} passos, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        else:
            resultado_instancia["DFS"] = {
                "solucao_encontrada": False,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"DFS: Solução não encontrada, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        
        # Busca Gulosa
        solucao, nos, tempo, memoria = busca_gulosa(instancia, estado_objetivo)
        if solucao:
            caminho = reconstruir_caminho(solucao)
            resultado_instancia["Gulosa"] = {
                "solucao_encontrada": True,
                "comprimento_caminho": len(caminho) - 1,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"Gulosa: Solução encontrada em {len(caminho) - 1} passos, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        else:
            resultado_instancia["Gulosa"] = {
                "solucao_encontrada": False,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"Gulosa: Solução não encontrada, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        
        # A*
        solucao, nos, tempo, memoria = busca_a_estrela(instancia, estado_objetivo)
        if solucao:
            caminho = reconstruir_caminho(solucao)
            resultado_instancia["A*"] = {
                "solucao_encontrada": True,
                "comprimento_caminho": len(caminho) - 1,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"A*: Solução encontrada em {len(caminho) - 1} passos, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        else:
            resultado_instancia["A*"] = {
                "solucao_encontrada": False,
                "nos_expandidos": nos,
                "tempo_execucao": tempo,
                "uso_memoria": memoria
            }
            print(f"A*: Solução não encontrada, {nos} nós expandidos, {tempo:.5f} segundos, {memoria:.2f} MB")
        
        resultados.append(resultado_instancia)
    
    return resultados


def gerar_estatisticas(resultados):
    """Gera estatísticas comparativas dos algoritmos"""
    
    algoritmos = ["BFS", "DFS", "Gulosa", "A*"]
    metricas = ["solucao_encontrada", "comprimento_caminho", "nos_expandidos", "tempo_execucao", "uso_memoria"]
    
    # Inicializa dicionário de estatísticas
    estatisticas = {algo: {metrica: [] for metrica in metricas} for algo in algoritmos}
    
    # Coleta as métricas para cada algoritmo
    for resultado in resultados:
        for algo in algoritmos:
            if algo in resultado:
                for metrica in metricas:
                    if metrica in resultado[algo]:
                        estatisticas[algo][metrica].append(resultado[algo][metrica])
    
    # Calcula médias
    medias = {algo: {} for algo in algoritmos}
    for algo in algoritmos:
        for metrica in metricas:
            if metrica == "solucao_encontrada":
                # Para solução encontrada, calculamos a porcentagem
                if estatisticas[algo][metrica]:
                    medias[algo][metrica] = sum(estatisticas[algo][metrica]) / len(estatisticas[algo][metrica]) * 100
                else:
                    medias[algo][metrica] = 0
            else:
                # Para outras métricas, calculamos a média considerando apenas quando a solução foi encontrada
                valores = []
                for i, val in enumerate(estatisticas[algo][metrica]):
                    if i < len(estatisticas[algo]["solucao_encontrada"]) and estatisticas[algo]["solucao_encontrada"][i]:
                        valores.append(val)
                
                if valores:
                    medias[algo][metrica] = sum(valores) / len(valores)
                else:
                    medias[algo][metrica] = None
    
    return medias


def main():
    # Define o estado objetivo (configuração final desejada)
    estado_objetivo = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
    
    # Carrega as instâncias do arquivo CSV
    instancias = []
    with open('ed02-puzzle8.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pula o cabeçalho
        
        for row in reader:
            # Converte as strings em números e as reorganiza em uma matriz 3x3
            nums = [int(n) for n in row]
            estado = [
                [nums[0], nums[1], nums[2]],
                [nums[3], nums[4], nums[5]],
                [nums[6], nums[7], nums[8]]
            ]
            instancias.append(np.array(estado))
    
    # Executa a comparação dos algoritmos
    resultados = comparar_algoritmos(instancias, estado_objetivo)
    
    # Gera estatísticas
    estatisticas = gerar_estatisticas(resultados)
    
    # Imprime tabela comparativa
    print("\n===== TABELA COMPARATIVA =====")
    print("Algoritmo | % Soluções | Passos Méd. | Nós Exp. Méd. | Tempo Méd. (s) | Memória Méd. (MB)")
    print("-" * 90)
    
    for algo in ["BFS", "DFS", "Gulosa", "A*"]:
        solucoes = estatisticas[algo]["solucao_encontrada"]
        passos = estatisticas[algo]["comprimento_caminho"]
        nos = estatisticas[algo]["nos_expandidos"]
        tempo = estatisticas[algo]["tempo_execucao"]
        memoria = estatisticas[algo]["uso_memoria"]
        
        passos_str = f"{passos:.2f}" if passos is not None else "N/A"
        nos_str = f"{nos:.2f}" if nos is not None else "N/A"
        tempo_str = f"{tempo:.5f}" if tempo is not None else "N/A"
        memoria_str = f"{memoria:.2f}" if memoria is not None else "N/A"
        
        print(f"{algo:9} | {solucoes:9.2f}% | {passos_str:11} | {nos_str:13} | {tempo_str:14} | {memoria_str:16}")
    
    print("\nConclusão:")
    print("Com base nos resultados, o algoritmo mais eficiente em termos de:")
    
    # Identifica o melhor algoritmo para cada métrica
    menor_passos = min(
        [algo for algo in ["BFS", "DFS", "Gulosa", "A*"] if estatisticas[algo]["comprimento_caminho"] is not None],
        key=lambda x: estatisticas[x]["comprimento_caminho"] if estatisticas[x]["comprimento_caminho"] is not None else float('inf')
    )
    
    menor_nos = min(
        [algo for algo in ["BFS", "DFS", "Gulosa", "A*"] if estatisticas[algo]["nos_expandidos"] is not None],
        key=lambda x: estatisticas[x]["nos_expandidos"] if estatisticas[x]["nos_expandidos"] is not None else float('inf')
    )
    
    menor_tempo = min(
        [algo for algo in ["BFS", "DFS", "Gulosa", "A*"] if estatisticas[algo]["tempo_execucao"] is not None],
        key=lambda x: estatisticas[x]["tempo_execucao"] if estatisticas[x]["tempo_execucao"] is not None else float('inf')
    )
    
    menor_memoria = min(
        [algo for algo in ["BFS", "DFS", "Gulosa", "A*"] if estatisticas[algo]["uso_memoria"] is not None],
        key=lambda x: estatisticas[x]["uso_memoria"] if estatisticas[x]["uso_memoria"] is not None else float('inf')
    )
    
    print(f"- Qualidade da solução (menor número de passos): {menor_passos}")
    print(f"- Nós expandidos (economia de processamento): {menor_nos}")
    print(f"- Tempo de execução: {menor_tempo}")
    print(f"- Uso de memória: {menor_memoria}")


if __name__ == "__main__":
    main()