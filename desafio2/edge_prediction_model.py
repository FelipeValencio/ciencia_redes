import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from haversine import haversine

# Load the graph
G = nx.read_gml("GraphMissingEdges.gml")

# Load the categories
categories = pd.read_csv("categories.csv")

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
    features["common_categories"] = len(
        set(G.nodes[node1]["categories"]).intersection(set(G.nodes[node2]["categories"])))
    features["review_diff"] = abs(G.nodes[node1]["reviewCount"] - G.nodes[node2]["reviewCount"])
    features["star_diff"] = abs(G.nodes[node1]["stars"] - G.nodes[node2]["stars"])
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

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Predict edges in edges_to_evaluate
evaluation_data = []
for _, row in edges_to_evaluate.iterrows():
    node1, node2 = row["venue1"], row["venue2"]
    features = extract_features(node1, node2)
    evaluation_data.append(features)

evaluation_df = pd.DataFrame(evaluation_data)
predictions = model.predict(evaluation_df)

# Prepare submission
edges_to_evaluate["link"] = predictions
edges_to_evaluate[["linkID", "link"]].to_csv("submission.csv", index=False)