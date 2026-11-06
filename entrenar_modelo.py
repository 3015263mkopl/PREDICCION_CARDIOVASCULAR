import json
import shutil
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ----------------------------------------------------------------------------
# 0. CONFIGURACION
# ----------------------------------------------------------------------------
SEMILLA = 42
URL_UCI = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNAS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
VAR_CONTINUAS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
VAR_CATEGORICAS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
COLUMNAS_X = VAR_CONTINUAS + VAR_CATEGORICAS

NOMBRES = {
    "age": "Edad",
    "sex": "Sexo",
    "cp": "Tipo de dolor toracico",
    "trestbps": "Presion arterial en reposo",
    "chol": "Colesterol serico",
    "fbs": "Glucemia en ayunas > 120",
    "restecg": "ECG en reposo",
    "thalach": "Frecuencia cardiaca maxima",
    "exang": "Angina por ejercicio",
    "oldpeak": "Depresion del ST",
    "slope": "Pendiente del ST",
    "ca": "Vasos coloreados",
    "thal": "Talasemia (talio)",
}

SALIDA = Path("salida_app")
SALIDA.mkdir(exist_ok=True)

print("=" * 70)
print("Version de scikit-learn:", sklearn.__version__)
print("=" * 70)

# ----------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA
# ----------------------------------------------------------------------------
df = pd.read_csv(URL_UCI, names=COLUMNAS, na_values="?")
df["target"] = (df["target"] > 0).astype(int)  # 0 = sin enfermedad, 1 = con enfermedad

n_original = len(df)
df_limpio = df.drop_duplicates().dropna().copy()
print(f"Registros originales: {n_original} | Registros limpios: {len(df_limpio)}")

# Todas las columnas como float: asi el pipeline guardado y la app reciben
# exactamente el mismo tipo de dato.
X = df_limpio[COLUMNAS_X].astype(float)
y = df_limpio["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEMILLA, stratify=y
)
print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

# ----------------------------------------------------------------------------
# 2. FIGURA DEL EDA
# ----------------------------------------------------------------------------
df_plot = df_limpio.copy()
df_plot["cp"] = df_plot["cp"].astype(int)
df_plot["Diagnostico"] = df_plot["target"].map({0: "Sin enfermedad", 1: "Con enfermedad"})
paleta = {"Sin enfermedad": "#3498db", "Con enfermedad": "#e74c3c"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.countplot(
    ax=axes[0, 0], data=df_plot, x="Diagnostico", hue="Diagnostico",
    order=["Sin enfermedad", "Con enfermedad"], palette=paleta, legend=False,
)
axes[0, 0].set_title("Distribucion de la variable objetivo")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Numero de pacientes")

sns.scatterplot(
    ax=axes[0, 1], data=df_plot, x="age", y="thalach", hue="Diagnostico",
    palette=paleta, alpha=0.85,
)
axes[0, 1].set_title("Edad vs frecuencia cardiaca maxima")
axes[0, 1].set_xlabel("Edad (anios)")
axes[0, 1].set_ylabel("Frecuencia cardiaca maxima (lpm)")

sns.countplot(
    ax=axes[1, 0], data=df_plot, x="cp", hue="Diagnostico",
    hue_order=["Sin enfermedad", "Con enfermedad"], palette=paleta,
)
axes[1, 0].set_title("Tipo de dolor toracico (cp) segun diagnostico")
axes[1, 0].set_xlabel("Tipo de dolor toracico (1 = tipica, 4 = asintomatico)")
axes[1, 0].set_ylabel("Numero de pacientes")

corr = df_limpio.corr()
mascara = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    ax=axes[1, 1], data=corr, mask=mascara, annot=True, fmt=".2f",
    cmap="vlag", vmin=-1, vmax=1, cbar=False,
)
axes[1, 1].set_title("Matriz de correlacion")

plt.tight_layout()
plt.savefig(SALIDA / "eda_figura.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------------
# 3. PIPELINE, MODELO BASE Y OPTIMIZACION
# ----------------------------------------------------------------------------
def crear_pipeline(clasificador):
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), VAR_CONTINUAS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), VAR_CATEGORICAS),
        ]
    )
    return Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", clasificador)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

# 3.1 Random Forest con parametros por defecto
pipe_base = crear_pipeline(RandomForestClassifier(random_state=SEMILLA))
recall_cv_base = cross_val_score(pipe_base, X_train, y_train, cv=cv, scoring="recall").mean()
pipe_base.fit(X_train, y_train)

# 3.2 Random Forest optimizado (rejilla moderada para que termine rapido)
param_grid = {
    "clasificador__n_estimators": [100, 200],
    "clasificador__max_depth": [None, 3, 5, 8],
    "clasificador__min_samples_split": [2, 5, 10],
    "clasificador__criterion": ["gini", "entropy"],
}
print("\nEjecutando GridSearchCV (96 combinaciones x 5 folds)...")
grid = GridSearchCV(
    crear_pipeline(RandomForestClassifier(random_state=SEMILLA)),
    param_grid, cv=cv, scoring="recall", n_jobs=-1,
)
grid.fit(X_train, y_train)
recall_cv_opt = grid.best_score_
print("Mejores parametros:", grid.best_params_)
print(f"Recall CV (base)       : {recall_cv_base:.3f}")
print(f"Recall CV (optimizado) : {recall_cv_opt:.3f}")

# 3.3 Seleccion del modelo final SOLO con validacion cruzada (no con el test)
if recall_cv_opt > recall_cv_base:
    modelo_final = grid.best_estimator_
    nombre_modelo = "Random Forest optimizado (GridSearchCV)"
    mejores_params = {k.replace("clasificador__", ""): v for k, v in grid.best_params_.items()}
else:
    modelo_final = pipe_base
    nombre_modelo = "Random Forest (parametros por defecto)"
    mejores_params = {"n_estimators": 100, "criterion": "gini", "max_depth": None,
                      "min_samples_split": 2}
print("\nModelo final seleccionado:", nombre_modelo)

# ----------------------------------------------------------------------------
# 4. EVALUACION EN EL CONJUNTO DE PRUEBA
# ----------------------------------------------------------------------------
y_pred = modelo_final.predict(X_test)
y_proba = modelo_final.predict_proba(X_test)[:, 1]

metricas_test = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
especificidad = tn / (tn + fp)

print("\n--- METRICAS EN PRUEBA ---")
for k, v in metricas_test.items():
    print(f"{k:10s}: {v:.3f}")
print(f"especificidad: {especificidad:.3f}")
print(f"Matriz de confusion -> TN={tn} FP={fp} FN={fn} TP={tp}")
print("\nReporte por clase:")
print(classification_report(y_test, y_pred, target_names=["Sin enfermedad", "Con enfermedad"]))

# Validacion cruzada sobre el entrenamiento (con puntaje de train para ver sobreajuste)
cv_res = cross_validate(
    clone(modelo_final), X_train, y_train, cv=cv, n_jobs=-1, return_train_score=True,
    scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall",
             "f1": "f1", "roc_auc": "roc_auc"},
)
metricas_cv = {}
print("--- VALIDACION CRUZADA (5 folds, solo entrenamiento) ---")
for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    val = cv_res[f"test_{m}"]
    tr = cv_res[f"train_{m}"]
    metricas_cv[m] = {
        "val_media": float(val.mean()),
        "val_std": float(val.std()),
        "train_media": float(tr.mean()),
    }
    print(f"{m:10s}: validacion {val.mean():.3f} +/- {val.std():.3f} | entrenamiento {tr.mean():.3f}")

# ----------------------------------------------------------------------------
# 5. FIGURAS DE EVALUACION
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["Sin enfermedad", "Con enfermedad"]).plot(
    ax=axes[0], cmap="Blues", values_format="d", colorbar=False
)
axes[0].set_title(f"Matriz de confusion (prueba, n={len(y_test)})")
axes[0].grid(False)

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color="darkorange", lw=2,
             label=f"Curva ROC (AUC = {metricas_test['roc_auc']:.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Azar")
axes[1].set_xlabel("Tasa de falsos positivos (1 - especificidad)")
axes[1].set_ylabel("Tasa de verdaderos positivos (recall)")
axes[1].set_title("Curva ROC")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig(SALIDA / "matriz_confusion_roc.png", dpi=150, bbox_inches="tight")
plt.show()

# Importancia por variable original (suma de las columnas one-hot de cada una)
prep = modelo_final.named_steps["preprocesador"]
nombres_salida = prep.get_feature_names_out()
importancias = modelo_final.named_steps["clasificador"].feature_importances_
acumulado = {}
for nombre, imp in zip(nombres_salida, importancias):
    base = nombre.split("__", 1)[1].split("_", 1)[0]
    acumulado[base] = acumulado.get(base, 0.0) + float(imp)

df_imp = (
    pd.DataFrame({"codigo": list(acumulado.keys()), "importancia": list(acumulado.values())})
    .assign(variable=lambda d: d["codigo"].map(NOMBRES))
    .sort_values("importancia", ascending=False)
    .reset_index(drop=True)
)
print("\n--- IMPORTANCIA POR VARIABLE ---")
print(df_imp[["variable", "importancia"]].round(3).to_string(index=False))

top10 = df_imp.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="importancia", y="variable", hue="variable",
            palette="viridis", legend=False)
plt.title("Top 10 variables clinicas por importancia (Random Forest)")
plt.xlabel("Importancia relativa (suma de las categorias de cada variable)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(SALIDA / "importancia_variables.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------------
# 6. REGISTROS CONOCIDOS DEL CONJUNTO DE PRUEBA (para verificar la app)
# ----------------------------------------------------------------------------
indices = []
primero_sano = y_test[y_test == 0].index[0]
primero_enfermo = y_test[y_test == 1].index[0]
indices += [primero_sano, primero_enfermo]
errados = X_test.index[(y_pred != y_test.to_numpy())]
if len(errados) > 0 and errados[0] not in indices:
    indices.append(errados[0])

ejemplos = []
print("\n--- REGISTROS DE PRUEBA PARA COMPARAR CON LA APP ---")
for idx in indices:
    fila = X_test.loc[[idx]]
    pred = int(modelo_final.predict(fila)[0])
    prob = float(modelo_final.predict_proba(fila)[0, 1])
    entradas = {c: float(fila.iloc[0][c]) for c in COLUMNAS_X}
    ejemplos.append({
        "id": int(idx),
        "descripcion": f"Registro {int(idx)} del conjunto de prueba",
        "entradas": entradas,
        "real": int(y_test.loc[idx]),
        "prediccion_notebook": pred,
        "probabilidad_notebook": prob,
    })
    print(f"Registro {idx}: real={int(y_test.loc[idx])} prediccion={pred} probabilidad={prob:.4f}")
    print("   ", entradas)

# ----------------------------------------------------------------------------
# 7. GUARDADO DE ARCHIVOS
# ----------------------------------------------------------------------------
joblib.dump(modelo_final, SALIDA / "modelo_pipeline.pkl", compress=3)

rangos = {c: [float(df_limpio[c].min()), float(df_limpio[c].max())] for c in VAR_CONTINUAS}

metricas = {
    "modelo": nombre_modelo,
    "criterio_seleccion": "Mayor recall medio en validacion cruzada de 5 folds (solo entrenamiento)",
    "recall_cv_base": float(recall_cv_base),
    "recall_cv_optimizado": float(recall_cv_opt),
    "mejores_parametros": mejores_params,
    "sklearn_version": sklearn.__version__,
    "n_original": int(n_original),
    "n_limpio": int(len(df_limpio)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "test": {k: float(v) for k, v in metricas_test.items()},
    "especificidad": float(especificidad),
    "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "cv": metricas_cv,
    "importancia_variables": [
        {"variable": r.variable, "importancia": float(r.importancia)}
        for r in df_imp.itertuples()
    ],
    "rangos_entrenamiento": rangos,
}
with open(SALIDA / "metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

with open(SALIDA / "ejemplos_test.json", "w", encoding="utf-8") as f:
    json.dump(ejemplos, f, ensure_ascii=False, indent=2)

with open(SALIDA / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(
        "streamlit>=1.40\n"
        "pandas\n"
        "numpy\n"
        f"scikit-learn=={sklearn.__version__}\n"
        "joblib\n"
    )

tabla = (
    f"Modelo final: {nombre_modelo}\n\n"
    "| Metrica | Prueba (n={nt}) | Validacion cruzada (media +/- desv.) |\n"
    "|---|---|---|\n"
).format(nt=len(X_test))
etiquetas = {"accuracy": "Exactitud", "precision": "Precision", "recall": "Recall",
             "f1": "F1-score", "roc_auc": "ROC-AUC"}
for m, etiqueta in etiquetas.items():
    tabla += (
        f"| {etiqueta} | {metricas_test[m]:.3f} | "
        f"{metricas_cv[m]['val_media']:.3f} +/- {metricas_cv[m]['val_std']:.3f} |\n"
    )
tabla += (
    f"\nMatriz de confusion en prueba: VN={tn}, FP={fp}, FN={fn}, VP={tp}. "
    f"Especificidad: {especificidad:.3f}.\n"
    f"Mejores parametros: {mejores_params}\n"
)
(SALIDA / "resultados.md").write_text(tabla, encoding="utf-8")
print("\n" + tabla)

# ----------------------------------------------------------------------------
# 8. EMPAQUETADO Y DESCARGA
# ----------------------------------------------------------------------------
shutil.make_archive("archivos_app", "zip", SALIDA)
print("Archivos generados en:", SALIDA.resolve())
for p in sorted(SALIDA.iterdir()):
    print("  -", p.name)

try:
    from google.colab import files  # solo existe en Colab
    files.download("archivos_app.zip")
except ImportError:
    print("\nNo se esta en Colab: tome el ZIP 'archivos_app.zip' de la carpeta actual.")
