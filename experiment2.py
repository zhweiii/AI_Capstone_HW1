import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("data/data.csv")
df["Conference"] = df["Conference"].map({"East": 0, "West": 1})

# Split data into train and test
train_data = df[df["SEASON"] < 2024]
test_data = df[df["SEASON"] >= 2024]

X = df.drop(columns=["Label", "SEASON", "TeamName"])
X_train = train_data.drop(columns=["Label", "SEASON", "TeamName"])
y_train = train_data["Label"]
X_test = test_data.drop(columns=["Label", "SEASON", "TeamName"])
y_test = test_data["Label"]

df_majority = train_data[train_data["Label"] == 0]
df_minority = train_data[train_data["Label"] == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,   
                                 n_samples=len(df_majority), 
                                 random_state=42)

train_data_balanced = pd.concat([df_majority, df_minority_upsampled])

X_train_balanced = train_data_balanced.drop(columns=["Label", "SEASON", "TeamName"])
y_train_balanced = train_data_balanced["Label"]

silhouette_scores = []  # List to store silhouette scores

for k in range(2, 30):
    kmeans = KMeans(n_clusters=k, random_state=42)  
    df["Cluster"] = kmeans.fit_predict(X)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, df["Cluster"])
    silhouette_scores.append(silhouette_avg)  # Append score for each k

    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 30), silhouette_scores, marker='o', color='skyblue', linestyle='-', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.grid(True)
plt.show()

# Plot result
feature1 = "REB"
feature2 = "PLUS_MINUS"
feature3 = "AST"

# Mark the points where label == 1
champions = df[df["Label"] == 1]

kmeans = KMeans(n_clusters=2, random_state=42)  
df["Cluster"] = kmeans.fit_predict(X)

# Assuming df and champions are already defined
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
scatter = ax.scatter(df[feature1], df[feature2], df[feature3], c=df["Cluster"], cmap="viridis", label="Teams")
ax.scatter(champions[feature1], champions[feature2], champions[feature3], c='red', label="Champions", edgecolors='w', s=100)
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel(feature3)
plt.title("NBA Team Clustering Based on Performance")
plt.colorbar(scatter, label="Cluster")
plt.legend()
plt.show()
