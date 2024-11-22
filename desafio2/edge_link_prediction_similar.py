import networkx as nx
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the graph
G = nx.read_gml("GraphMissingEdges.gml")

# Load edges to evaluate
edges_to_evaluate = pd.read_csv("edgesToEvaluate.csv")


# Helper function to calculate similarity scores
def extract_similarity_features(G, node1, node2):
    features = {}
    features["common_neighbors"] = common_neighbors_score(G, node1, node2)
    features["jaccard_coefficient"] = jaccard_coefficient_score(G, node1, node2)
    features["resource_allocation_index"] = resource_allocation_index_score(G, node1, node2)
    # Katz similarity is not added here due to its computational complexity, can be added if needed
    return features


# Prepare training data by taking existing edges and generating non-edges
edges = list(G.edges())
non_edges = list(nx.non_edges(G))
np.random.shuffle(non_edges)
non_edges = non_edges[:len(edges)]  # Balance the dataset

data = []
labels = []
for edge in edges:
    node1, node2 = edge
    features = extract_similarity_features(G, node1, node2)
    data.append(features)
    labels.append(1)  # Edge exists

for edge in non_edges:
    node1, node2 = edge
    features = extract_similarity_features(G, node1, node2)
    data.append(features)
    labels.append(0)  # Edge does not exist

# Create DataFrame
df = pd.DataFrame(data)
X = df
y = labels

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Predict edges in edges_to_evaluate
evaluation_data = []
for _, row in edges_to_evaluate.iterrows():
    node1, node2 = row["venue1"], row["venue2"]
    features = extract_similarity_features(G, node1, node2)
    evaluation_data.append(features)

evaluation_df = pd.DataFrame(evaluation_data)
predictions = model.predict(evaluation_df)

# Prepare submission
edges_to_evaluate["link"] = predictions
edges_to_evaluate[["linkID", "link"]].to_csv("submission.csv", index=False)