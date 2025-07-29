import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
CARPETA = "df_EEG/PRE"
META_PATH = os.path.join(CARPETA, "meta.csv")
COLUMNA_Y = "y"
OUTPUT_CSV = "zonas_y_features_sin_normalizacion.csv"

# === ZONAS CEREBRALES ===
zonas = {
    "frontal": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["POz", "O1", "Oz", "O2"],
    "izquierda": ["Fp1","AF3","F7", "F3","FC5", "FC1","T7", "C3","CP5", "CP1","P7", "P3", "O1"],
    "derecha": ["Fp2","AF4","F4", "F8","FC2", "FC6","C4", "T8","CP2", "CP6","P4", "P8", "O2"],
    "derecha_frontal": ["Fp2","AF4","F4", "F8","FC2", "FC6"],
    "izquierda_frontal": ["Fp1","AF3","F7", "F3","FC5", "FC1"],
    "full": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
        "POz", "O1", "Oz", "O2" ]
}


# === FEATURES AGRUPADAS ===
grupos_features = {
{
    "temp": ["_mean", "_std", "_max",  "_min", "_range"],
    "diff": ["_d1_mean", "_d1_std", "_d2_mean", "_d2_std"],
    "form": ["_arc_len", "_integral","_skew",  "_kurt"],
    "rel":  [ "_rms",  "_area_perim", "_energy_perim"],
    "energy": ["_welch_abs", "_welch_rel"],
    
}}

modelos = {
    "lr": (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
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
        RandomForestClassifier(class_weight="balanced", random_state=42),
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
            "clf__n_neighbors": [1, 3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]  # Manhattan (p=1) y Euclídea (p=2)
        }
    ),
    "xgb": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.04 , 0.05, 0.06, 0.07],
            "clf__max_depth": [3, 5, 7, 9],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0]
        }
    ),
    "et": (
        ExtraTreesClassifier(class_weight="balanced", random_state=42),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    ),
    
    "gb": (
        GradientBoostingClassifier(random_state=42),
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
    MLPClassifier(max_iter=1000, random_state=42),
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
    CatBoostClassifier(verbose=0, random_state=42),
    {
        "clf__iterations": [100, 200],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__depth": [3, 5, 7]
    }
)

}


scoring = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "accuracy": make_scorer(accuracy_score),
    "precision_macro": make_scorer(precision_score, average="macro"),
    "recall_macro": make_scorer(recall_score, average="macro"),
    "roc_auc": make_scorer(roc_auc_score, multi_class="ovo")
}

meta = pd.read_csv(META_PATH, index_col=0)
meta.index = meta.index.astype(str)

filas = []
archivos = [f for f in os.listdir(CARPETA) if f.startswith("X_")]

for archivo in archivos:
    print(f"📂 Evaluando archivo: {archivo}")
    try:
        df = pd.read_csv(os.path.join(CARPETA, archivo))
        df["ID"] = df["ID"].astype(str)

        groups = df["ID"]
        y = groups.map(meta[COLUMNA_Y]).astype("Int64")
        mask = y.notna()
        X = df.drop(columns=["ID", "tipo", "ventana"], errors="ignore")
        X, y, groups = X[mask], y[mask], groups[mask]

        for zona, canales in zonas.items():
            cols_zona = [c for c in X.columns if any(c.startswith(e + "_") for e in canales)]
            if not cols_zona:
                print(f"⚠️ Zona {zona} vacía en {archivo}")
                continue
            X_zona = X[cols_zona]
            for nombre, (modelo, grid_params) in modelos.items():
                print(f"   🔍 Modelo: {nombre} - Zona: {zona}")
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", modelo)
                ])
                gkf = GroupKFold(n_splits=5)
                grid = GridSearchCV(
                    pipeline, grid_params, scoring=scoring, refit="accuracy",
                    cv=gkf.split(X_zona, y, groups), n_jobs=-1, return_train_score=True
                )
                grid.fit(X_zona, y)
                scores = grid.cv_results_
                fila = {
                    "archivo": archivo, "zona": zona, "modelo": nombre,
                    "f1_macro": scores["mean_test_f1_macro"][grid.best_index_],
                    "accuracy": scores["mean_test_accuracy"][grid.best_index_],
                    "precision_macro": scores["mean_test_precision_macro"][grid.best_index_],
                    "recall_macro": scores["mean_test_recall_macro"][grid.best_index_],
                    "roc_auc": scores["mean_test_roc_auc"][grid.best_index_],
                    "train_f1_macro": scores["mean_train_f1_macro"][grid.best_index_],
                    "gap_f1": scores["mean_train_f1_macro"][grid.best_index_] - grid.best_score_,
                    "params": grid.best_params_
                }
                filas.append(fila)

        for nombre_feat, sufijo in grupos_features.items():
            cols_feat = [c for c in X.columns if c.endswith(sufijo)]
            if not cols_feat:
                continue
            X_feat = X[cols_feat]
            for nombre, (modelo, grid_params) in modelos.items():
                print(f"   🔍 Modelo: {nombre} - Feature: {nombre_feat}")
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", modelo)
                ])
                grid = GridSearchCV(
                    pipeline, grid_params, scoring=scoring, refit="f1_macro",
                    cv=GroupKFold(n_splits=5).split(X_feat, y, groups),
                    n_jobs=-1, return_train_score=True
                )
                grid.fit(X_feat, y)
                scores = grid.cv_results_
                fila = {
                    "archivo": archivo, "feat": nombre_feat, "modelo": nombre,
                    "f1_macro": grid.best_score_,
                    "accuracy": scores["mean_test_accuracy"][grid.best_index_],
                    "precision_macro": scores["mean_test_precision_macro"][grid.best_index_],
                    "recall_macro": scores["mean_test_recall_macro"][grid.best_index_],
                    "roc_auc": scores["mean_test_roc_auc"][grid.best_index_],
                    "train_f1_macro": scores["mean_train_f1_macro"][grid.best_index_],
                    "gap_f1": scores["mean_train_f1_macro"][grid.best_index_] - grid.best_score_,
                    "params": grid.best_params_
                }
                filas.append(fila)
    except Exception as e:
        print(f"❌ Error con {archivo}: {e}")

# === GUARDAR ===
df_final = pd.DataFrame(filas).sort_values("accuracy", ascending=False)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Resultados guardados en {OUTPUT_CSV}")
