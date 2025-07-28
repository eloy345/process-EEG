import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
ARCHIVO = "X_delta_VB.csv"
CARPETA = "df_EEG/PRE"
META_PATH = os.path.join(CARPETA, "meta.csv")
RUTA_X = os.path.join(CARPETA, ARCHIVO)

# === MEJOR MODELO ===
modelo = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=1.0,
    eval_metric="logloss"
)

# === CARGAR DATOS ===
meta = pd.read_csv(META_PATH, index_col=0)
meta.index = meta.index.astype(str)
df = pd.read_csv(RUTA_X)
groups = df["ID"].astype(str)
df = df.drop(columns=["ID", "tipo", "ventana"], errors="ignore")
y = groups.map(meta["y"])
mask = y.notna()
X = df[mask]
y = y[mask]
groups = groups[mask]

# === PIPELINE ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.9, random_state=42)),
    ("clf", modelo)
])

gkf = GroupKFold(n_splits=5)

from sklearn.model_selection import cross_validate

scores = cross_validate(
    pipeline, X, y, cv=gkf.split(X, y, groups),
    scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovo"],
    return_estimator=True
)
print("\n📊 Resultados de validación cruzada:")
print(f"{'F1 macro:':<15} {scores['test_f1_macro'].mean():.4f}")
print(f"{'Accuracy:':<15} {scores['test_accuracy'].mean():.4f}")
print(f"{'Precision:':<15} {scores['test_precision_macro'].mean():.4f}")
print(f"{'Recall:':<15} {scores['test_recall_macro'].mean():.4f}")
print(f"{'ROC AUC (OVO):':<15} {scores['test_roc_auc_ovo'].mean():.4f}")

# === PREDICCIONES Y MATRIZ DE CONFUSIÓN ===
y_pred = cross_val_predict(pipeline, X, y, cv=gkf.split(X, y, groups), groups=groups)

print("\n🔎 Classification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matriz de confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
