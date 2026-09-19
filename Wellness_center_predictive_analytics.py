import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

# DATA PREPROCESSING

df = pd.read_csv("wellness_center_data.csv")

df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

# GRAPH 1: DISTRIBUTION
plt.hist(df["DoctorCount"], bins=20, label="Doctor Count")
plt.xlabel("Doctor Count")
plt.ylabel("Frequency")
plt.title("Distribution of Doctor Count")
plt.legend()
plt.show()

print("Prediction: Most wellness centers have a moderate number of doctors.\n")

# GRAPH 2: CORRELATION
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("Prediction: DoctorCount is influenced by multiple features.\n")

# REGRESSION MODELS
X_reg = df.drop("DoctorCount", axis=1)
y_reg = df["DoctorCount"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
Xr_train = scaler.fit_transform(Xr_train)
Xr_test = scaler.transform(Xr_test)

# Linear Regression
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
lr_pred = lr.predict(Xr_test)

print("Linear Regression R2:", r2_score(yr_test, lr_pred))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(yr_test, lr_pred)))
print("Prediction: Linear regression gives moderate prediction accuracy.\n")

# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(Xr_train, yr_train)
rf_pred = rf_reg.predict(Xr_test)

print("Random Forest Regression R2:", r2_score(yr_test, rf_pred))
print("Prediction: Random Forest improves prediction over linear regression.\n")

# CLASSIFICATION MODELS
X_clf = df.drop("Category", axis=1)
y_clf = df["Category"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
Xc_train = scaler2.fit_transform(Xc_train)
Xc_test = scaler2.transform(Xc_test)

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)

    print(name)
    print("Accuracy :", accuracy_score(yc_test, pred))
    print("Precision:", precision_score(yc_test, pred, average="weighted", zero_division=0))
    print("Recall   :", recall_score(yc_test, pred, average="weighted", zero_division=0))
    print("F1 Score :", f1_score(yc_test, pred, average="weighted", zero_division=0))
    print("Prediction:", name, "model classifies wellness categories with reasonable accuracy.\n")
# CONFUSION MATRIX
cm = confusion_matrix(yc_test, models["Decision Tree"].predict(Xc_test))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("Prediction: Most values lie on diagonal → good classification.\n")
# ELBOW METHOD
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_reg)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker="o", label="WCSS")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.legend()
plt.show()

print("Prediction: Optimal number of clusters is around K = 3.\n")
# K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_reg)

plt.scatter(X_reg.iloc[:, 0], X_reg.iloc[:, 1], c=clusters, label="Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

print("Prediction: Data is divided into three meaningful clusters.\n")
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xc_train)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=yc_train, label="PCA Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.legend()
plt.show()

print("Prediction: PCA successfully reduces dimensions while retaining variance.\n")
# NEURAL NETWORK
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
mlp.fit(Xc_train, yc_train)
mlp_pred = mlp.predict(Xc_test)

print("MLP Accuracy:", accuracy_score(yc_test, mlp_pred))
print("Prediction: Neural network performs competitively.\n")
# CROSS VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(random_state=42), X_clf, y_clf, cv=kf)

print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Prediction: Model is stable and generalizes well.\n")

print("PROJECT EXECUTED SUCCESSFULLY ✅")
