import networkx as nx
import pandas as pd
import numpy as np
import cupy as cp
import networkit as nk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carrega o grafo (representa os locais e conexões entre eles)
G = nx.read_gml("GraphMissingEdges.gml")

# Carrega os pares de nós (arestas) que devem ser avaliados
edges_to_evaluate = pd.read_csv("edgesToEvaluate.csv")

# Cria um mapeamento dos identificadores dos nós para índices (necessário para certas operações matriciais)
node_to_index = {node: index for index, node in enumerate(G.nodes())}
index_to_node = {index: node for index, node in enumerate(G.nodes())}

# Calcula a pontuação de vizinhos comuns entre dois nós
# Estratégia: O número de vizinhos em comum fornece uma ideia da proximidade estrutural dos nós no grafo.
def common_neighbors_score(G, node1, node2):
    common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))
    return common_neighbors


# Calcula o coeficiente de Jaccard entre dois nós
# Estratégia: Mede a similaridade entre os vizinhos de dois nós. A relação é entre o tamanho da interseção dos vizinhos e o tamanho da união dos vizinhos.
def jaccard_coefficient_score(G, node1, node2):
    union_neighbors = set(G.neighbors(node1)).union(set(G.neighbors(node2)))
    if len(union_neighbors) == 0:
        return 0  # Evitar divisão por zero
    intersection_neighbors = set(G.neighbors(node1)).intersection(set(G.neighbors(node2)))
    return len(intersection_neighbors) / len(union_neighbors)


# Calcula a pontuação de Katz entre dois nós
# Estratégia: Baseia-se na contagem ponderada do número de caminhos entre dois nós, penalizando caminhos mais longos.
def katz_score(G, node1, node2, beta=0.005):
    # Converte a matriz de adjacência do grafo em uma matriz densa
    A = nx.adjacency_matrix(G).todense()
    # Converte para uma matriz na GPU usando a biblioteca CuPy
    A_gpu = cp.array(A, dtype=cp.float32)
    I = cp.eye(A.shape[0])  # Matriz identidade
    katz_similarity = cp.linalg.inv(I - beta * A_gpu) - I  # Fórmula do scoring Katz

    # Mapeia os nós para índices correspondentes na matriz
    idx1 = node_to_index[node1]
    idx2 = node_to_index[node2]

    return katz_similarity[idx1, idx2]  # Retorna a similaridade para o par de nós


# Calcula o índice de alocação de recursos (Resource Allocation Index)
# Estratégia: Quanto menor o grau dos vizinhos comuns de dois nós, maior o peso atribuído à relação entre eles.
def resource_allocation_index_score(G, node1, node2):
    ra_index = sum(1 / G.degree(n) for n in nx.common_neighbors(G, node1, node2))
    return ra_index


# Calcula o índice Adamic-Adar entre dois nós
# Estratégia: Parecido com o índice de alocação de recursos, mas utiliza o log do grau do nó para ponderação.
def adamic_adar_index_score(G, node1, node2):
    score = 0.0
    for n in nx.common_neighbors(G, node1, node2):
        degree = G.degree(n)
        if degree > 1:  # Garante que o grau seja maior que 1 para evitar log(1) = 0
            score += 1 / np.log(degree)
    return score


# Função auxiliar para calcular todas as pontuações de similaridade para um par de nós
# Estratégia: Utiliza várias medidas de proximidade no grafo para capturar relações estruturais entre os nós.
def extract_similarity_features(G, node1, node2):
    features = {}
    features["common_neighbors"] = common_neighbors_score(G, node1, node2)
    features["jaccard_coefficient"] = jaccard_coefficient_score(G, node1, node2)
    # A pontuação de Katz não estava convergindo devido à alta complexidade computacional (O(n^3)),
    # mas pode ser reativada se necessário.
    # features["katz_score"] = katz_score(G, node1, node2)
    features["resource_allocation_index"] = resource_allocation_index_score(G, node1, node2)
    features["adamic_adar_index"] = adamic_adar_index_score(G, node1, node2)
    return features


# Preparação dos dados de treinamento: utiliza arestas existentes no grafo e arestas fictícias
# Estratégia: Treinar o modelo com exemplos de arestas presentes ("1") e ausentes ("0").
edges = list(G.edges())  # Arestas existentes
non_edges = list(nx.non_edges(G))  # Arestas que não existem
np.random.shuffle(non_edges)
non_edges = non_edges[:len(edges)]  # Balanceia o dataset (número de não-arestas == número de arestas)

data = []
labels = []
for edge in edges:
    node1, node2 = edge
    features = extract_similarity_features(G, node1, node2)  # Extrai as características de similaridade
    data.append(features)
    labels.append(1)  # Marca como "1" (com aresta)

for edge in non_edges:
    node1, node2 = edge
    features = extract_similarity_features(G, node1, node2)
    data.append(features)
    labels.append(0)  # Marca como "0" (sem aresta)

# Cria um DataFrame para organizar os dados
df = pd.DataFrame(data)
X = df  # Dados de características
y = labels  # Rótulos (se existe ou não uma aresta)

# Divide os dados entre treinamento (80%) e teste (20%)
# Estratégia: Manter um conjunto de dados separado para avaliar o desempenho do modelo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina um modelo de Regressão Logística
# Estratégia: Modelo simples e eficiente para problemas binários (nesse caso, prever se existe uma conexão entre dois nós).
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Avalia o modelo
# Estratégia: Usa a acurácia para medir a proporção de previsões corretas no conjunto de teste.
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Realiza as previsões para os pares de nós a serem avaliados
evaluation_data = []
for _, row in edges_to_evaluate.iterrows():
    node1, node2 = row["venue1"], row["venue2"]
    features = extract_similarity_features(G, node1, node2)
    evaluation_data.append(features)

evaluation_df = pd.DataFrame(evaluation_data)
predictions = model.predict(evaluation_df)  # Previsões para os pares avaliados

# Prepara o arquivo de submissão
edges_to_evaluate["link"] = predictions  # Adiciona as previsões
edges_to_evaluate[["linkID", "link"]].to_csv("submission_similar.csv", index=False)
