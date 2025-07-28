import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
CARPETA = "df_EEG/PRE"
META_PATH = os.path.join(CARPETA, "meta.csv")
COLUMNA_Y = "y"
OUTPUT_CSV = "resultados_grid_search_zonas.csv"

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
    "mean": "_mean",
    "std": "_std",
    "max": "_max",
    "min": "_min",
    "range": "_range",
    "d1_mean": "_d1_mean",
    "d1_std": "_d1_std",
    "d2_mean": "_d2_mean",
    "d2_std": "_d2_std",
    "arc_len": "_arc_len",
    "rms": "_rms",
    "area_perim": "_area_perim",
    "energy_perim": "_energy_perim",
    "integral": "_integral",
    "skew": "_skew",
    "kurt": "_kurt",
    "welch_abs": "_welch_abs",
    "welch_rel": "_welch_rel"
}



# === DEFINIR MODELOS Y GRIDS ===
modelos = {
    "lr": (
        LogisticRegression(class_weight="balanced", max_iter=1000),
        {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"]
        }
    ),
    "svm": (
        SVC(kernel="linear", class_weight="balanced", probability=True),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    ),
    "rf": (
        RandomForestClassifier(class_weight="balanced"),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [1, 3, 5, 7],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        }
    ),
    "xgb": (
        XGBClassifier(eval_metric="logloss"),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.03, 0.05],
            "clf__max_depth": [3, 5],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0]
        }
    ),
    "et": (
        ExtraTreesClassifier(class_weight="balanced"),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    ),
    "gb": (
        GradientBoostingClassifier(),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.05],
            "clf__max_depth": [3, 5]
        }
    ),
    "lgbm": (
        LGBMClassifier(),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.05],
            "clf__max_depth": [3, 5, -1],
            "clf__num_leaves": [15, 31]
        }
    ),
    "qda": (
        QuadraticDiscriminantAnalysis(),
        {"clf__reg_param": [0.0, 0.01, 0.1]}
    ),
    "mlp": (
        MLPClassifier(max_iter=1000),
        {
            "clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__activation": ["relu", "tanh"]
        }
    ),
    "ada": (
        AdaBoostClassifier(),
        {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.01, 0.1, 1.0]
        }
    ),
    "catboost": (
        CatBoostClassifier(verbose=0),
        {
            "clf__iterations": [100, 200],
            "clf__learning_rate": [0.01, 0.05],
            "clf__depth": [3, 5]
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

# === CARGAR META ===
meta = pd.read_csv(META_PATH, index_col=0)
meta.index = meta.index.astype(str)

filas = []
archivos = [f for f in os.listdir(CARPETA) if f.endswith("_VB.csv") and f.startswith("X_")]

for archivo_vb in archivos:
    banda = archivo_vb.replace("_VB.csv", "")
    archivo_basal = f"{banda}.csv"
    ruta_vb = os.path.join(CARPETA, archivo_vb)
    ruta_basal = os.path.join(CARPETA, archivo_basal)

    print(f"📂 Evaluando archivo: {archivo_vb} con normalización usando {archivo_basal}")
    try:
        df_vb = pd.read_csv(ruta_vb)
        df_basal = pd.read_csv(ruta_basal)

        df_vb["ID"] = df_vb["ID"].astype(str)
        df_basal["ID"] = df_basal["ID"].astype(str)

        # Normalizar VB / basal intra-sujeto
        #X_norm = []
   #     for sujeto in df_vb["ID"].unique():
  #          datos_vb = df_vb[df_vb["ID"] == sujeto].drop(columns=["ID", "tipo", "ventana"], errors="ignore")
   #         datos_basal = df_basal[df_basal["ID"] == sujeto].drop(columns=["ID", "tipo", "ventana"], errors="ignore")
 #           if not datos_basal.empty and not datos_vb.empty:
   #             media_basal = datos_basal.mean()
   #             norm = datos_vb / media_basal.replace(0, np.nan)
  #              norm["ID"] = sujeto
  #              X_norm.append(norm)

   #     if not X_norm:
   #         print(f"⚠️ No se pudo normalizar ningún sujeto para {archivo_vb}")
   #         continue
#
    #    df = pd.concat(X_norm, ignore_index=True).dropna(axis=1, how="any")
        groups = df["ID"]
        df = df.drop(columns=["ID"])
        #df = df.loc[:, df.columns.str.contains("mean|^ID$")]  # Seleccionar columnas 

        y = groups.map(meta[COLUMNA_Y]).astype("Int64")
        mask = y.notna()
        X = df[mask]
        y = y[mask]
        groups = groups[mask]

        for zona, canales in zonas.items():
            cols_zona = [c for c in X.columns if any(c.startswith(elec + "_") for elec in canales)]
            if not cols_zona:
                print(f"⚠️ Zona {zona} vacía en {archivo_vb}")
                continue
            X_zona = X[cols_zona]

            for nombre, (modelo, grid_params) in modelos.items():
                print(f"   🔍 Modelo: {nombre} - Zona: {zona}_ Archivo {archivo_vb}")
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    #("selectk", SelectKBest(score_func=f_classif, k=20)),
                    #("pca", PCA(n_components=0.95, random_state=42)),
                    ("clf", modelo)
                ])
                gkf = GroupKFold(n_splits=5)
                grid = GridSearchCV(
                    pipeline, grid_params,
                    scoring=scoring, refit="f1_macro",
                    cv=gkf.split(X_zona, y, groups),
                    n_jobs=-1, return_train_score=True
                )
                warnings.filterwarnings("ignore")
                grid.fit(X_zona, y)
                best = grid.best_params_
                scores = grid.cv_results_
                fila = {
                    "archivo": archivo_vb,
                    "zona": zona,
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

        # --- FEATURES ---
        for nombre_feat, sufijo in grupos_features.items():
            columnas_feat = [c for c in X.columns if c.endswith(sufijo)]
            if not columnas_feat:
                    continue
            X_original = X.copy()
            X_feat = X_original[columnas_feat]
            for nombre, (modelo, grid_params) in modelos.items():
                print(f"   🔍 Modelo: {nombre} - Feature: {nombre_feat}_ Archivo {archivo_vb}")
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    #("selectk", SelectKBest(score_func=f_classif, k=20)),
                    #("pca", PCA(n_components=0.95, random_state=42)),
                    ("clf", modelo)
                ])
                gkf = GroupKFold(n_splits=5)
                grid = GridSearchCV(
                    pipeline, grid_params,
                    scoring=scoring, refit="f1_macro",
                    cv=gkf.split(X_feat, y, groups),
                    n_jobs=-1, return_train_score=True
                )
                grid.fit(X_feat, y)
                warnings.filterwarnings("ignore")
                best = grid.best_params_
                scores = grid.cv_results_
                fila = {
                    "archivo": archivo_vb,
                    "feat": nombre_feat,
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
        print(f"❌ Error con {archivo_vb}: {e}")



# === GUARDAR RESULTADOS ===
df_final = pd.DataFrame(filas)
df_final = df_final.sort_values("f1_macro", ascending=False)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Resultados guardados en {OUTPUT_CSV}")
