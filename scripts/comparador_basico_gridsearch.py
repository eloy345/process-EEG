
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")


# === CONFIG ===
CARPETA = "df_EEG/PRE"
META_PATH = "df_EEG/PRE/meta.csv"
COLUMNA_Y = "y"
OUTPUT_CSV = "resultados_grid_search_reinicio.csv"

# === CARGAR META ===
meta = pd.read_csv(META_PATH, index_col=0)
meta.index = meta.index.astype(str)

# === DEFINIR MODELOS Y GRIDS ===
modelos = {
    "lr": (
        LogisticRegression(class_weight="balanced", max_iter=1000),
        {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l1", "l2"],  # podrías incluir "l1" si usas solver='liblinear'
            "clf__solver": ["liblinear", "saga"]
        }
    ),
    "svm": (
        SVC(kernel="linear", class_weight="balanced", probability=True),
        {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100]
        }
    ),
    "rf": (
        RandomForestClassifier(class_weight="balanced"),
        {
            "clf__n_estimators": [100, 200, 300, 400, 500],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]  # Manhattan (p=1) y Euclídea (p=2)
        }
    ),
    "xgb": (
        XGBClassifier(eval_metric="logloss"),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.03, 0.04 , 0.05, 0.06, 0.07],
            "clf__max_depth": [3, 5, 7, 9],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0]
        }
    ),
    "et": (
        ExtraTreesClassifier(class_weight="balanced"),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    ),
    
    "gb": (
        GradientBoostingClassifier(),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [3, 5, 7]
        }
    ),
    
    #"lgbm": (
     #   LGBMClassifier(),
      #  {
       #     "clf__n_estimators": [100, 200, 300],
        #    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
         #   "clf__max_depth": [3, 5, 7, -1],
          #  "clf__num_leaves": [15, 31, 63]
    #    }
   # ),
    
    "qda": (
        QuadraticDiscriminantAnalysis(),
    {
        "clf__reg_param": [0.0, 0.01, 0.1, 0.5]
    }
),
    "mlp": (
    MLPClassifier(max_iter=1000),
    {
        "clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "clf__alpha": [0.0001, 0.001, 0.005, 0.01, 0.02 , 0.03],
        "clf__activation": ["relu", "tanh"]
    }
),
    
    "ada": (
    AdaBoostClassifier(),
    {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 1.0]
    }
),
    "catboost": (
    CatBoostClassifier(verbose=0),
    {
        "clf__iterations": [100, 200],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__depth": [3, 5, 7]
    }
)

}

# === MÉTRICAS ===
scoring = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "accuracy": make_scorer(accuracy_score),
    "precision_macro": make_scorer(precision_score, average="macro"),
    "recall_macro": make_scorer(recall_score, average="macro"),
    "roc_auc": make_scorer(roc_auc_score, multi_class="ovo")
}

# === EVALUACIÓN ===
filas = []
archivos = [f for f in os.listdir(CARPETA) if f.endswith(".csv") and f.startswith("X_")]

for archivo in archivos:
    ruta = os.path.join(CARPETA, archivo)
    print(f"📂 Evaluando archivo: {archivo}")
    try:
        df = pd.read_csv(ruta)
        groups = df["ID"].astype(str)
        df = df.drop(columns=["ID", "tipo", "ventana"], errors="ignore")
        df = df.loc[:, df.columns.str.contains("F|^ID$")]  # Seleccionar columnas 
        y = groups.map(meta[COLUMNA_Y])
        mask = y.notna()
        X = df[mask]
        y = y[mask]
        groups = groups[mask]

        for nombre, (modelo, grid_params) in modelos.items():
            print(f"   🔍 Modelo: {nombre} (GridSearchCV)")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                #("selectk", SelectKBest(score_func=f_classif, k=100)),
                #("pca", PCA(n_components=0.9, random_state=42)),
                ("clf", modelo)
            ])
            gkf = GroupKFold(n_splits=5)
            grid = GridSearchCV(
                pipeline, grid_params,
                scoring=scoring, refit="accuracy",
                cv=gkf.split(X, y, groups),
                n_jobs=-1, return_train_score=True
            )
            grid.fit(X, y)
            best = grid.best_params_
            scores = grid.cv_results_
            fila = {
                "archivo": archivo,
                "modelo": nombre,
                "f1_macro": grid.best_score_,
                "accuracy": scores["mean_test_accuracy"][grid.best_index_],
                "precision_macro": scores["mean_test_precision_macro"][grid.best_index_],
                "recall_macro": scores["mean_test_recall_macro"][grid.best_index_],
                "roc_auc": scores["mean_test_roc_auc"][grid.best_index_],
                "train_f1_macro": scores["mean_train_f1_macro"][grid.best_index_],
                "gap_f1": scores["mean_train_f1_macro"][grid.best_index_] - grid.best_score_,
                "params": best
            }
            filas.append(fila)
    except Exception as e:
        print(f"⚠️ Error con archivo {archivo}: {e}")

# === GUARDAR RESULTADOS ===
df_final = pd.DataFrame(filas)
df_final = df_final.sort_values("accuracy", ascending=False)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Resultados guardados en {OUTPUT_CSV}")
