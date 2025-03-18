import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns

def convert_season_format(season):
    start_year, _ = season.split("-")
    end_year = int(start_year) + 1
    return end_year

df = pd.read_csv("data/data.csv")

df["Conference"] = df["Conference"].map({"East": 0, "West": 1})
df["SEASON_END_YEAR"] = df["SEASON"].apply(convert_season_format)

train_data = df[df["SEASON_END_YEAR"] < 2022]
test_data = df[df["SEASON_END_YEAR"] >= 2022]

X_train = train_data.drop(columns=["Label", "SEASON", "SEASON_END_YEAR", "TeamName"])
y_train = train_data["Label"]
X_test = test_data.drop(columns=["Label", "SEASON", "SEASON_END_YEAR", "TeamName"])
y_test = test_data["Label"]

# normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


# # SVM
# svm_model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight=class_weights_dict)
# svm_model.fit(X_train_scaled, y_train)
# svm_preds = svm_model.predict(X_test_scaled)

# predictions = pd.DataFrame({"SEASON_END_YEAR": test_data["SEASON_END_YEAR"], 
#                             "Team": test_data["TeamName"], 
#                             "Conference": test_data["Conference"].map({0: "East", 1: "West"}),
#                             "Label": test_data["Label"],
#                             "Prediction": svm_preds})

# print("SVM prediction:")
# print(predictions)

# print("---- SVM result:")

# y_true = predictions["Label"]
# y_pred = predictions["Prediction"]

# accuracy = accuracy_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred, average="weighted")
# f1 = f1_score(y_true, y_pred, average="weighted")

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# conf_matrix = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(7,5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# # Random Forest
# rf_model = RandomForestClassifier(n_estimators = 200, min_samples_split = 2,
#                                   max_features = 'log2', max_depth = 40, 
#                                   random_state=42, class_weight=class_weights_dict)
# rf_model.fit(X_train_scaled, y_train)
# rf_preds = rf_model.predict(X_test_scaled)

# predictions_rf = pd.DataFrame({
#     "SEASON_END_YEAR": test_data["SEASON_END_YEAR"],
#     "Team": test_data["TeamName"],
#     "Conference": test_data["Conference"].map({0: "East", 1: "West"}),
#     "Label": test_data["Label"],
#     "Prediction": rf_preds
# })

# print("Random Forest prediction:")
# print(predictions_rf)

# # evaluate
# accuracy_rf = accuracy_score(y_test, rf_preds)
# recall_rf = recall_score(y_test, rf_preds, average="weighted")
# f1_rf = f1_score(y_test, rf_preds, average="weighted")

# print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
# print(f"Random Forest Recall: {recall_rf:.4f}")
# print(f"Random Forest F1 Score: {f1_rf:.4f}")

# conf_matrix_rf = confusion_matrix(y_test, rf_preds)

# plt.figure(figsize=(7,5))
# sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens",
#             xticklabels=["Champion", "Runner-up"], 
#             yticklabels=["Champion", "Runner-up"])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Random Forest Confusion Matrix")
# plt.show()

# feature_importance = rf_model.feature_importances_
# feature_names = X_train.columns

# importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
# importance_df = importance_df.sort_values(by="Importance", ascending=False)

# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance in Random Forest")
# plt.show()

# XGBoost 
sample_weights = np.array([class_weights_dict[label] for label in y_train])
lr = 0.008
n_estimators = 600
depth = 25
xgb_model = XGBClassifier(n_estimators=n_estimators, learning_rate=lr, 
                          max_depth=depth, random_state=42, eval_metric="mlogloss")
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
xgb_preds = xgb_model.predict(X_test)

#show prediction
predictions_xgb = pd.DataFrame({
    "SEASON_END_YEAR": test_data["SEASON_END_YEAR"],
    "Team": test_data["TeamName"],
    "Conference": test_data["Conference"].map({0: "East", 1: "West"}),
    "Label": test_data["Label"],
    "Prediction": xgb_preds
})

print("XGBoost champ prediction:")
print(predictions_xgb[predictions_xgb["Prediction"] == 0])

# evaluate
print("--- XGBoost result:")
print(f"lr: {lr}, n_estimators: {n_estimators}, max_depth: {depth}")
accuracy = accuracy_score(y_test, xgb_preds)
recall = recall_score(y_test, xgb_preds, average="weighted")
f1 = f1_score(y_test, xgb_preds, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
conf_matrix = confusion_matrix(y_test, xgb_preds)
print(conf_matrix)

# plt.figure(figsize=(7,5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# # feature importance
# feature_importance = xgb_model.feature_importances_
# feature_names = X_train.columns

# importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
# importance_df = importance_df.sort_values(by="Importance", ascending=False)

# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", data=importance_df)
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance in XGBoost")
# plt.show()




# # Support Vector Machine (SVM) with class weights
# svm_model = SVC(kernel='rbf', C=3.0, gamma='auto', class_weight=class_weights_dict, probability=True)
# svm_model.fit(X_train_scaled, y_train)
# svm_probs = svm_model.predict_proba(X_test_scaled)
# # print("SVM prediction:", svm_probs)

# test_data["SVM_Prob"] = svm_probs[:, 0]
# svm_champions = test_data.loc[test_data.groupby("SEASON")["SVM_Prob"].idxmax()]

# print("SVM champ prediction:")
# print(svm_champions[["SEASON","TeamName", "SVM_Prob"]])

# XGBoost model
# xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# xgb_model.fit(X_train, y_train)
# xgb_probs = xgb_model.predict_proba(X_test)
# test_data["XGB_Prob"] = xgb_probs[:, 0]
# xgb_champions = test_data.loc[test_data.groupby("SEASON")["XGB_Prob"].idxmax()]

# print("XGBoost champ prediction:")
# print(xgb_champions[["SEASON", "XGB_Prob"]])

# Random Forest evaluate
# rf_preds = rf_model.predict(X_test_scaled)
# print("Random Forest result:")
# print("Accuracy:", accuracy_score(y_test, rf_preds))
# print(confusion_matrix(y_test, rf_preds))
# print(classification_report(y_test, rf_preds))


# SVM evaluate
# svm_preds = svm_model.predict(X_test_scaled)
# print("SVM result:")
# print("Accuracy:", accuracy_score(y_test, svm_preds))
# print(confusion_matrix(y_test, svm_preds))
# print(classification_report(y_test, svm_preds))

# XGBoost evaluate
# xgb_preds = xgb_model.predict(X_test)
# print("XGBoost result:")
# print("Accuracy:", accuracy_score(y_test, xgb_preds))
# print(confusion_matrix(y_test, xgb_preds))
# print(classification_report(y_test, xgb_preds))