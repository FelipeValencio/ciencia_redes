import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from haversine import haversine
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Carrega o grafo
G = nx.read_gml("GraphMissingEdges.gml")

# Carrega as categorias das lojas
categories = pd.read_csv("categories.csv")
categories_dict = categories.set_index("CategoryId")["names"].to_dict()

# Carrega as arestas a serem avaliadas (onde precisamos prever se existe ou não uma conexão entre duas lojas)
edges_to_evaluate = pd.read_csv("edgesToEvaluate.csv")


# Função auxiliar para calcular a distância geográfica entre dois nós
# Utiliza as coordenadas de latitude e longitude armazenadas no grafo
def calculate_distance(node1, node2):
    coord1 = (G.nodes[node1]["latitude"], G.nodes[node1]["longitude"])
    coord2 = (G.nodes[node2]["latitude"], G.nodes[node2]["longitude"])
    return haversine(coord1, coord2)

# Engenharia de Features (características utilizadas para modelagem)
def extract_features(node1, node2):
    features = {}
    # Calcula a distância geográfica entre os dois nós fornecidos
    features["distance"] = calculate_distance(node1, node2)

    # Decodifica as categorias associadas a cada nó
    node1_categories = set(G.nodes[node1]["categories"].split(','))
    node2_categories = set(G.nodes[node2]["categories"].split(','))

    # Calcula o número de categorias em comum entre os dois nós
    features["common_categories"] = len(node1_categories.intersection(node2_categories))

    # Adiciona o número total de categorias para cada nó
    features["categories_1"] = len(node1_categories)
    features["categories_2"] = len(node2_categories)

    # Cria variáveis binárias (one-hot encoding) para cada categoria.
    # Por exemplo, se o nó pertence à categoria "bar", definimos "bar" = 1, outras categorias como 0 e assim por diante.
    for cat_id, cat_name in categories_dict.items():
        features[f"cat_{cat_id}_1"] = 1 if str(cat_id) in node1_categories else 0
        features[f"cat_{cat_id}_2"] = 1 if str(cat_id) in node2_categories else 0

    # Calcula a diferença absoluta no número de avaliações ("reviewCount") atribuídas aos dois estabelecimentos
    review_count_1 = int(G.nodes[node1]["reviewCount"])
    review_count_2 = int(G.nodes[node2]["reviewCount"])
    features["review_diff"] = abs(review_count_1 - review_count_2)

    # Calcula a diferença absoluta entre as notas ("stars") atribuídas aos dois estabelecimentos
    stars_1 = float(G.nodes[node1]["stars"])
    stars_2 = float(G.nodes[node2]["stars"])
    features["star_diff"] = abs(stars_1 - stars_2)

    return features


# Preparação dos dados de treinamento
edges = list(G.edges(data=True)) # Obtém todas as arestas existentes no grafo
data = []
labels = []
for edge in edges:
    node1, node2, attr = edge
    # Extrai as features para os pares de nós já conectados (arestas existentes no grafo)
    features = extract_features(node1, node2)
    features["weight"] = attr["weight"] # Adiciona o peso da aresta
    data.append(features)
    labels.append(1) # Marca como 1 se existe uma aresta

# Gera exemplos negativos (pares de nós sem conexões) para ensinar ao modelo o que significa "ausência de aresta"
nodes = list(G.nodes())
for _ in range(len(edges)): # Gera o mesmo número de exemplos negativos que positivos (balanceamento do dataset)
    while True:
        node1, node2 = np.random.choice(nodes, 2, replace=False) # Escolhe dois nós aleatórios
        if not G.has_edge(node1, node2): # Garante que não exista uma conexão entre os dois
            features = extract_features(node1, node2)
            features["weight"] = 0 # Define peso 0 para arestas inexistentes
            data.append(features)
            labels.append(0)  # Marca como 0 definindo que não existe uma aresta
            break

# Cria um DataFrame para organizar os dados
df = pd.DataFrame(data)
X = df.drop(columns=["weight"])
y = labels

# Divide os dados em treino (80%) e teste (20%) para avaliação do desempenho do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados para remover escalas diferentes entre as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treina e avalia diferentes modelos de machine learning
models = {
    "Logistic Regression": LogisticRegression(random_state=42),  # Modelo simples de regressão para previsões binárias
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), # Combina várias árvores de decisão (robusto a overfitting)
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42), # Técnica de boosting que otimiza gradualmente as previsões
    # "SVM": SVC(kernel='linear', random_state=42), # Tentei utilizar SVM mas nao estava convergindo :(
    "Neural Network": MLPClassifier(random_state=42, max_iter=300) # Rede neural multicamada
}

# Avalia o desempenho de cada modelo utilizando métricas de acurácia e F1-score
for name, model in models.items():
    model.fit(X_train_scaled, y_train) # Treina o modelo nos dados escalados
    y_pred = model.predict(X_test_scaled) # Realiza previsões no conjunto de teste
    accuracy = accuracy_score(y_test, y_pred) # Calcula a acurácia (proporção de previsões corretas)
    f1 = f1_score(y_test, y_pred) # Calcula o F1-score, melhor para a previsão do Kaggle
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} F1-score: {f1}")

# Realiza previsões nos pares de nós a serem avaliados
evaluation_data = []
for _, row in edges_to_evaluate.iterrows():
    node1, node2 = row["venue1"], row["venue2"]
    features = extract_features(node1, node2) # Extrai as mesmas features usadas nos dados de treino
    evaluation_data.append(features)

# Normaliza os dados dos pares de teste e realiza a predição
evaluation_df = pd.DataFrame(evaluation_data)
evaluation_df_scaled = scaler.transform(evaluation_df)
predictions = models["Random Forest"].predict(evaluation_df_scaled)   # Selecionamos o modelo que apresentou o melhor desempenho

# Prepara o arquivo de submissão com as previsões
edges_to_evaluate["link"] = predictions
edges_to_evaluate[["linkID", "link"]].to_csv("submission_ML.csv", index=False)