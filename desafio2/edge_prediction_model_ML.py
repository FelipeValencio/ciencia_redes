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
from sklearn.svm import SVC

# Load the graph
G = nx.read_gml("GraphMissingEdges.gml")

# Load the categories
categories = pd.read_csv("categories.csv")
categories_dict = categories.set_index("CategoryId")["names"].to_dict()

# Load edges to evaluate
edges_to_evaluate = pd.read_csv("edgesToEvaluate.csv")


# Helper function to calculate geographical distance
def calculate_distance(node1, node2):
    coord1 = (G.nodes[node1]["latitude"], G.nodes[node1]["longitude"])
    coord2 = (G.nodes[node2]["latitude"], G.nodes[node2]["longitude"])
    return haversine(coord1, coord2)

# Feature Engineering
def extract_features(node1, node2):
    features = {}
    features["distance"] = calculate_distance(node1, node2)

    # Decode the category IDs for each node
    node1_categories = set(G.nodes[node1]["categories"].split(','))
    node2_categories = set(G.nodes[node2]["categories"].split(','))

    features["common_categories"] = len(node1_categories.intersection(node2_categories))

    # Add amount of categories
    features["categories_1"] = len(node1_categories)
    features["categories_2"] = len(node2_categories)

    # Add one hot encoding per category. If store contains bar, its "bar" = 1 and other categories = 0
    for cat_id, cat_name in categories_dict.items():
        features[f"cat_{cat_id}_1"] = 1 if str(cat_id) in node1_categories else 0
        features[f"cat_{cat_id}_2"] = 1 if str(cat_id) in node2_categories else 0

    # Convert reviewCount from string to integer
    review_count_1 = int(G.nodes[node1]["reviewCount"])
    review_count_2 = int(G.nodes[node2]["reviewCount"])
    features["review_diff"] = abs(review_count_1 - review_count_2)

    # Convert stars from string to float
    stars_1 = float(G.nodes[node1]["stars"])
    stars_2 = float(G.nodes[node2]["stars"])
    features["star_diff"] = abs(stars_1 - stars_2)

    return features


# Prepare training data
edges = list(G.edges(data=True))
data = []
labels = []
for edge in edges:
    node1, node2, attr = edge
    features = extract_features(node1, node2)
    features["weight"] = attr["weight"]
    data.append(features)
    labels.append(1)  # Edge exists

# Generate non-existent edges for negative examples
nodes = list(G.nodes())
for _ in range(len(edges)):
    while True:
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(node1, node2):
            features = extract_features(node1, node2)
            features["weight"] = 0
            data.append(features)
            labels.append(0)  # Edge does not exist
            break

# Create DataFrame
df = pd.DataFrame(data)
X = df.drop(columns=["weight"])
y = labels

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate different models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    # "SVM": SVC(kernel='linear', random_state=42),
    "Neural Network": MLPClassifier(random_state=42, max_iter=300)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} F1-score: {f1}")

# Predict edges in edges_to_evaluate
evaluation_data = []
for _, row in edges_to_evaluate.iterrows():
    node1, node2 = row["venue1"], row["venue2"]
    features = extract_features(node1, node2)
    evaluation_data.append(features)

evaluation_df = pd.DataFrame(evaluation_data)
evaluation_df_scaled = scaler.transform(evaluation_df)
predictions = models["Random Forest"].predict(evaluation_df_scaled)  # Choose the best performing model

# Prepare submission
edges_to_evaluate["link"] = predictions
edges_to_evaluate[["linkID", "link"]].to_csv("submission_ML.csv", index=False)

## Média de 0.72 de accuracy