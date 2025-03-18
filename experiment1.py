import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

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

# Random Forest Regressor
rf_model = RandomForestRegressor(max_depth=10, n_estimators=300, criterion='friedman_mse',
                                max_features='sqrt', random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_probs = rf_model.predict(X_test)

# Add predicted probabilities to test data
test_data = test_data.copy()  
test_data.loc[:, "RF_Prob"] = rf_probs
rf_champions = test_data.loc[test_data.groupby("SEASON")["RF_Prob"].idxmax()]
print("--- Random Forest ---")
print("Highest Champion Prob:")
print(rf_champions[["SEASON", "TeamName", "RF_Prob"]])

# evaluation
mse = mean_squared_error(y_test, rf_probs)

# # Cross-validation
split = 10
kf = KFold(n_splits=split, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mse_cv_scores = cross_val_score(rf_model, X_train_balanced, y_train_balanced, cv=kf, scoring=mse_scorer)

# Plot cross-validated MSE
plt.figure(figsize=(10, 6))
plt.plot(range(1, split + 1), -mse_cv_scores, marker='o', linestyle='-', color='b', label='MSE')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('Score')
plt.margins(y=0.4)
plt.title('evalution for each fold')
plt.grid(True)
plt.show()

# Feature Importance
feature_importance = rf_model.feature_importances_
feature_names = X_train_balanced.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Print evaluation
print("evaluation:")
print(f"MSE: {mse:.4f}")
print(f"Cross-validated MSE: {-mse_cv_scores.mean():.4f}")

# SVR
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale')
svr_model.fit(X_train, y_train)
svr_probs = svr_model.predict(X_test)
svr_probs = np.tanh(svr_probs)

# print("svr_probs:", svr_probs)

# Add predicted probabilities to test data
test_data = test_data.copy()  
test_data.loc[:, "SVR_Prob"] = svr_probs
rf_champions = test_data.loc[test_data.groupby("SEASON")["SVR_Prob"].idxmax()]
print("--- SVR ---")
print("Highest Champion Prob:")
print(rf_champions[["SEASON", "TeamName", "SVR_Prob"]])

# Cross-validation
split = 10
kf = KFold(n_splits=split, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mse_cv_scores = cross_val_score(svr_model, X_train, y_train, cv=kf, scoring=mse_scorer)

# Plot cross-validated MSE
plt.figure(figsize=(10, 6))
plt.plot(range(1, split + 1), -mse_cv_scores, marker='o', linestyle='-', color='b', label='MSE')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('evalution for each fold')
plt.margins(x=0.2, y=0.2) 
plt.show()

# evaluation
mse_svr = mean_squared_error(y_test, svr_probs)

# Print evaluation
print("evaluation:")
print(f"MSE: {mse_svr:.4f}")
print(f"Cross-validated MSE: {-mse_cv_scores.mean():.4f}")

# Feature Importance
result = permutation_importance(svr_model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(np.array(X_train.columns)[sorted_idx], result.importances_mean[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("SVR Feature Importance (Permutation Importance)")
plt.gca().invert_yaxis()
plt.show()
