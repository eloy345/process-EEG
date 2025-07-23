import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# 1. Cargar datos
X = pd.read_csv("df_EEG/PRE/X_gamma.csv", index_col=0)
meta = pd.read_csv("df_EEG/PRE/meta.csv", index_col=0)

# 3. Cruzar IDs
ids_comunes = X.index.intersection(meta.index)
X = X.loc[ids_comunes]
y = meta.loc[ids_comunes]["y_recommended"]

# 4. Normalización y PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)  
X_pca = pca.fit_transform(X_scaled)
print(f"🔷 PCA redujo de {X.shape[1]} a {X_pca.shape[1]} dimensiones")

# 5. Definir modelos y grids
modelos = {
    "RandomForest": (RandomForestClassifier(), {
        "clf__n_estimators": [50, 100,200 ],
        "clf__max_depth": [None, 5, 10,20 ]
    }),
    "SVM": (SVC(probability=True), {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"]
    }),
    "KNN": (KNeighborsClassifier(), {
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"]
    }),
    "LogReg": (LogisticRegression(max_iter=1000), {
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }),
    "XGBoost": (XGBClassifier(eval_metric='logloss'), {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.1, 0.01]
    })
}

# 6. Evaluar cada modelo
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nombre, (modelo, grid_params) in modelos.items():
    print(f"\n🔹 {nombre}")
    pipeline = Pipeline([
        ("clf", modelo)
    ])
    grid = GridSearchCV(pipeline, param_grid=grid_params, cv=skf, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_pca, y)
    
    print("🔹 Mejor F1 (val cruzada):", grid.best_score_)
    print("🔹 Mejores parámetros:", grid.best_params_)
    
    # Evaluación final
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()
