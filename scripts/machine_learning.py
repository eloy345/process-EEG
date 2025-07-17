import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# 1. Cargar datos
X = pd.read_csv("df_EEG/PRE/X_gamma.csv", index_col=0)
df_meta = pd.read_csv("df_EEG/PRE/meta.csv", index_col=0)

# 1.1 Cruzar solo los IDs que están en ambos
ids_validos = X.index.intersection(df_meta.index)
X = X.loc[ids_validos]
y = df_meta.loc[ids_validos]["y_recommended"]

# 2. Filtrar solo columnas numéricas
X = X.select_dtypes(include="number")


# 3. Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Aplicar PCA 
pca = PCA(n_components=0.9, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"➡️ PCA redujo de {X.shape[1]} a {X_pca.shape[1]} dimensiones")

# 5. GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svm = SVC(class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_pca, y)

print("🧪 Mejor combinación de parámetros:", grid_search.best_params_)
print("✅ F1-score medio (CV):", grid_search.best_score_)



# 6. División train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# 7. Modelo SVM
clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 8. Validación cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_pca, y, cv=skf, scoring='f1_macro')
print("F1-score medio (validación cruzada):", scores.mean())

# 9. Evaluar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()