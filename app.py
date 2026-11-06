import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

COLUMNAS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Rangos que admite el formulario (tambien se usan para validar en el servidor)
RANGOS_FORMULARIO = {
    "age": (20, 100),
    "trestbps": (90, 200),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 6.5),
}
ETIQUETAS = {
    "age": "Edad",
    "trestbps": "Presión arterial en reposo",
    "chol": "Colesterol sérico",
    "thalach": "Frecuencia cardíaca máxima",
    "oldpeak": "Depresión del ST",
}

# ----------------------------------------------------------------------------
# CONFIGURACION DE LA PAGINA Y ESTILOS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción Cardiovascular",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.8rem; color: #C0392B; text-align: center; font-weight: bold; }
    .sub-title { font-size: 1.2rem; text-align: center; color: #7F8C8D; margin-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load(BASE / "modelo_pipeline.pkl")
    except Exception as e:
        st.error(
            "No se pudo cargar 'modelo_pipeline.pkl'. Verifique que el archivo esté en el "
            "repositorio y que la versión de scikit-learn de requirements.txt sea la misma "
            f"con la que se entrenó. Detalle: {e}"
        )
        return None


@st.cache_data
def cargar_json(nombre):
    ruta = BASE / nombre
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def mostrar_imagen(nombre, descripcion):
    ruta = BASE / nombre
    if not ruta.exists():
        st.warning(f"Falta el archivo '{nombre}' en el repositorio.")
        return
    try:
        st.image(str(ruta), caption=descripcion, width="stretch")
    except Exception:
        # Versiones antiguas de Streamlit
        st.image(str(ruta), caption=descripcion, use_container_width=True)


def construir_dataframe(datos):
    """Misma estructura y tipo (float) que se usó al entrenar."""
    return pd.DataFrame([datos], columns=COLUMNAS).astype(float)


def validar_entradas(datos, rangos_train):
    """Devuelve (errores, advertencias). Los errores bloquean la predicción."""
    errores, advertencias = [], []

    for campo, (minimo, maximo) in RANGOS_FORMULARIO.items():
        valor = datos.get(campo)
        nombre = ETIQUETAS[campo]
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            errores.append(f"{nombre}: falta el valor.")
        elif not (minimo <= valor <= maximo):
            errores.append(f"{nombre}: {valor} está fuera del rango permitido ({minimo} a {maximo}).")

    if errores:
        return errores, advertencias

    if datos["thalach"] > 220 - datos["age"]:
        advertencias.append(
            "La frecuencia cardíaca máxima supera la estimación teórica (220 - edad). "
            "Verifique el dato."
        )
    if datos["chol"] > 400:
        advertencias.append("Colesterol muy elevado (> 400 mg/dl). Verifique el dato.")
    if datos["trestbps"] >= 180:
        advertencias.append("Presión arterial en reposo muy alta (>= 180 mm Hg). Verifique el dato.")

    if rangos_train:
        for campo, (minimo, maximo) in rangos_train.items():
            if campo in datos and not (minimo <= datos[campo] <= maximo):
                advertencias.append(
                    f"{ETIQUETAS[campo]} ({datos[campo]}) queda fuera del rango observado en el "
                    f"entrenamiento ({minimo:g} a {maximo:g}). La predicción es una extrapolación "
                    "y es menos confiable."
                )
    return errores, advertencias


def mostrar_resultado(prediccion, probabilidad):
    col_a, col_b = st.columns(2)
    with col_a:
        if prediccion == 1:
            st.error(
                "### Riesgo alto\nEl modelo estima una **alta probabilidad** de presencia de "
                "enfermedad cardiovascular (umbral de decisión: 50 %)."
            )
        else:
            st.success(
                "### Riesgo bajo\nEl modelo estima una **baja probabilidad** de presencia de "
                "enfermedad cardiovascular (umbral de decisión: 50 %)."
            )
    with col_b:
        st.metric("Probabilidad estimada de enfermedad", f"{probabilidad * 100:.1f} %")
        st.progress(float(probabilidad))


modelo = cargar_modelo()
metricas = cargar_json("metricas.json")
ejemplos = cargar_json("ejemplos_test.json")
rangos_train = (metricas or {}).get("rangos_entrenamiento")

# ----------------------------------------------------------------------------
# ENCABEZADO Y PRESENTACION
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">Sistema de Predicción Cardiovascular</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Herramienta analítica basada en Machine Learning '
    "(Dataset Heart Disease de Cleveland - UCI)</p>",
    unsafe_allow_html=True,
)

st.warning(
    "**Aviso importante:** esta aplicación es un **prototipo académico**. Las predicciones "
    "generadas no sustituyen en ningún caso el diagnóstico, consejo o tratamiento de un "
    "profesional de la salud."
)

with st.expander("Proyecto, integrantes y fuente de los datos", expanded=True):
    st.write(
        """
**Objetivo:** desarrollar un modelo predictivo capaz de estimar la probabilidad de presencia de
enfermedad cardiovascular a partir de variables clínicas y demográficas del paciente.

**Integrantes del grupo:**
* Héctor Cervantes Echeverria
* Juan Guillermo Sarmiento Garrido
* Santiago Valle Palacio
* Diego Mendoza Gutiérrez

**Fuente de los datos:** base de datos *Heart Disease* del repositorio UCI Machine Learning
Repository (subconjunto de Cleveland, 303 registros originales).
Enlace: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
    )

tab1, tab2, tab3 = st.tabs(
    ["Realizar predicción", "Exploración de datos (EDA)", "Modelo y evaluación"]
)

# ----------------------------------------------------------------------------
# PESTANA 1: FORMULARIO Y PREDICCION
# ----------------------------------------------------------------------------
with tab1:
    if modelo is None:
        st.error("El modelo no está disponible, por lo que no se pueden generar predicciones.")
    else:
        st.markdown("### Ingrese los datos clínicos del paciente")

        with st.form("formulario_prediccion"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Datos demográficos y básicos**")
                age = st.number_input("Edad (años)", min_value=20, max_value=100, value=50, step=1)
                sex = st.selectbox(
                    "Sexo", options=[1, 0], format_func=lambda x: "Hombre" if x == 1 else "Mujer"
                )
                trestbps = st.number_input(
                    "Presión arterial en reposo (mm Hg)", min_value=90, max_value=200, value=120
                )
                chol = st.number_input(
                    "Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200
                )
                fbs = st.selectbox(
                    "Glucemia en ayunas > 120 mg/dl",
                    options=[0, 1],
                    format_func=lambda x: "Sí" if x == 1 else "No",
                )

            with col2:
                st.markdown("**Datos clínicos y de esfuerzo**")
                cp = st.selectbox(
                    "Tipo de dolor torácico",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "Angina típica",
                        2: "Angina atípica",
                        3: "Dolor no anginoso",
                        4: "Asintomático",
                    }[x],
                )
                restecg = st.selectbox(
                    "Resultado del electrocardiograma en reposo",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "Anormalidad de la onda ST-T",
                        2: "Hipertrofia ventricular izquierda",
                    }[x],
                )
                thalach = st.number_input(
                    "Frecuencia cardíaca máxima alcanzada (lpm)",
                    min_value=60, max_value=220, value=150,
                )
                exang = st.selectbox(
                    "Angina inducida por ejercicio",
                    options=[1, 0],
                    format_func=lambda x: "Sí" if x == 1 else "No",
                )
                oldpeak = st.number_input(
                    "Depresión del ST inducida por ejercicio (mm)",
                    min_value=0.0, max_value=6.5, value=1.0, step=0.1,
                )

            st.markdown("**Otros estudios**")
            col3, col4, col5 = st.columns(3)
            with col3:
                slope = st.selectbox(
                    "Pendiente del segmento ST",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "Ascendente", 2: "Plana", 3: "Descendente"}[x],
                )
            with col4:
                ca = st.selectbox("Número de vasos principales coloreados", options=[0, 1, 2, 3])
            with col5:
                thal = st.selectbox(
                    "Resultado de la gammagrafía con talio",
                    options=[3, 6, 7],
                    format_func=lambda x: {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"}[x],
                )

            enviar = st.form_submit_button("Calcular riesgo cardiovascular")

        if enviar:
            datos = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            errores, advertencias = validar_entradas(datos, rangos_train)

            if errores:
                st.error("No se puede calcular la predicción. Corrija los siguientes datos:")
                for e in errores:
                    st.write(f"- {e}")
            else:
                for a in advertencias:
                    st.warning(a)

                entrada = construir_dataframe(datos)
                prediccion = int(modelo.predict(entrada)[0])
                probabilidad = float(modelo.predict_proba(entrada)[0][1])

                st.divider()
                st.markdown("### Resultado del análisis para este paciente")
                mostrar_resultado(prediccion, probabilidad)

                texto_importancia = ""
                if metricas and metricas.get("importancia_variables"):
                    top = [v["variable"] for v in metricas["importancia_variables"][:4]]
                    texto_importancia = (
                        " Globalmente, las variables que más pesan en el modelo son: "
                        + ", ".join(top)
                        + "."
                    )
                st.info(
                    "**Cómo leer este resultado:** la probabilidad proviene de un modelo "
                    "Random Forest entrenado con pacientes del conjunto de Cleveland. Es una "
                    "estimación estadística para este conjunto de valores, no un diagnóstico. "
                    "La importancia de las variables es **global** (aplica al modelo completo) "
                    "y no explica por separado a este paciente." + texto_importancia
                )

        # Verificacion contra el notebook
        st.divider()
        with st.expander("Verificación con un registro conocido del conjunto de prueba"):
            st.write(
                "Seleccione un registro del conjunto de prueba. La aplicación calcula la "
                "predicción con el modelo guardado y la compara con la obtenida en el notebook "
                "para el mismo registro."
            )
            if not ejemplos:
                st.info("No se encontró 'ejemplos_test.json' en el repositorio.")
            else:
                opciones = {
                    f"{e['descripcion']} (diagnóstico real: "
                    f"{'con enfermedad' if e['real'] == 1 else 'sin enfermedad'})": e
                    for e in ejemplos
                }
                elegido = opciones[st.selectbox("Registro", list(opciones.keys()))]
                st.dataframe(pd.DataFrame([elegido["entradas"]]), hide_index=True)

                if st.button("Calcular y comparar con el notebook"):
                    entrada = construir_dataframe(elegido["entradas"])
                    pred_app = int(modelo.predict(entrada)[0])
                    prob_app = float(modelo.predict_proba(entrada)[0][1])

                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Notebook: probabilidad",
                        f"{elegido['probabilidad_notebook'] * 100:.2f} %",
                    )
                    c2.metric("Aplicación: probabilidad", f"{prob_app * 100:.2f} %")

                    coincide = (
                        pred_app == elegido["prediccion_notebook"]
                        and abs(prob_app - elegido["probabilidad_notebook"]) < 1e-6
                    )
                    if coincide:
                        st.success(
                            "La predicción y la probabilidad de la aplicación coinciden con las "
                            "del notebook."
                        )
                    else:
                        st.error(
                            "No coinciden. Revise que 'modelo_pipeline.pkl' y 'ejemplos_test.json' "
                            "provengan de la misma ejecución del script de entrenamiento."
                        )

# ----------------------------------------------------------------------------
# PESTANA 2: EDA
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### Exploración de datos (EDA)")
    st.write(
        "Análisis exploratorio realizado sobre los registros depurados del dataset de Cleveland "
        "(valores faltantes y duplicados eliminados)."
    )
    mostrar_imagen("eda_figura.png", "EDA: distribución del objetivo, edad vs frecuencia cardíaca, dolor torácico y correlaciones")

    st.markdown("#### Hallazgos principales")
    st.markdown(
        """
* **Balance de clases:** el conjunto está razonablemente balanceado (160 pacientes sin
  enfermedad y 137 con enfermedad), por lo que el accuracy no está distorsionado por una clase
  dominante.
* **Edad y frecuencia cardíaca máxima:** la frecuencia máxima tiende a disminuir con la edad
  (correlación de -0,39), y los pacientes con enfermedad se concentran en valores más bajos.
* **Tipo de dolor torácico:** la categoría 4 (asintomático) concentra la mayoría de los casos con
  enfermedad (103 frente a 39 sin enfermedad). Es un patrón contraintuitivo que conviene
  interpretar con cuidado.
* **Correlaciones con el objetivo:** las más altas corresponden a la talasemia (0,53), el número
  de vasos coloreados (0,46), la angina por ejercicio y la depresión del ST (0,42 cada una) y la
  frecuencia cardíaca máxima (-0,42). La glucemia en ayunas y el colesterol casi no se
  relacionan linealmente con el objetivo.
"""
    )

    st.markdown("#### Diccionario de variables")
    diccionario = pd.DataFrame(
        [
            ("age", "Edad", "años"),
            ("sex", "Sexo", "1 = hombre, 0 = mujer"),
            ("cp", "Tipo de dolor torácico", "1 = angina típica, 2 = angina atípica, 3 = no anginoso, 4 = asintomático"),
            ("trestbps", "Presión arterial en reposo", "mm Hg"),
            ("chol", "Colesterol sérico", "mg/dl"),
            ("fbs", "Glucemia en ayunas > 120 mg/dl", "1 = sí, 0 = no"),
            ("restecg", "ECG en reposo", "0 = normal, 1 = anormalidad ST-T, 2 = hipertrofia ventricular izquierda"),
            ("thalach", "Frecuencia cardíaca máxima", "latidos por minuto"),
            ("exang", "Angina inducida por ejercicio", "1 = sí, 0 = no"),
            ("oldpeak", "Depresión del ST inducida por ejercicio", "mm"),
            ("slope", "Pendiente del segmento ST", "1 = ascendente, 2 = plana, 3 = descendente"),
            ("ca", "Vasos principales coloreados por fluoroscopía", "0 a 3"),
            ("thal", "Gammagrafía con talio", "3 = normal, 6 = defecto fijo, 7 = defecto reversible"),
            ("target", "Presencia de enfermedad (variable objetivo)", "0 = no, 1 = sí (original 1 a 4 agrupado)"),
        ],
        columns=["Variable", "Descripción", "Unidad o categorías"],
    )
    st.dataframe(diccionario, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# PESTANA 3: MODELO Y EVALUACION
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Rendimiento del modelo")

    if not metricas:
        st.warning(
            "No se encontró 'metricas.json'. Ejecute 'entrenar_modelo.py' y suba el archivo "
            "generado al repositorio para mostrar las métricas reales."
        )
    else:
        st.write(
            f"**Modelo final:** {metricas['modelo']}. "
            f"**Criterio de selección:** {metricas['criterio_seleccion']}. "
            "En un contexto médico se prioriza el **recall** (sensibilidad) para reducir los "
            "falsos negativos."
        )

        t = metricas["test"]
        st.markdown(f"#### Resultados en el conjunto de prueba (n = {metricas['n_test']})")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Exactitud", f"{t['accuracy'] * 100:.1f} %")
        m2.metric("Recall (sensibilidad)", f"{t['recall'] * 100:.1f} %")
        m3.metric("Precisión", f"{t['precision'] * 100:.1f} %")
        m4.metric("F1-score", f"{t['f1'] * 100:.1f} %")
        m5.metric("ROC-AUC", f"{t['roc_auc']:.3f}")

        cmx = metricas["matriz_confusion"]
        st.write(
            f"**Matriz de confusión:** {cmx['tp']} enfermos detectados (verdaderos positivos), "
            f"{cmx['fn']} enfermos no detectados (falsos negativos), {cmx['fp']} sanos marcados "
            f"como enfermos (falsos positivos) y {cmx['tn']} sanos correctamente identificados "
            f"(verdaderos negativos). Especificidad: {metricas['especificidad'] * 100:.1f} %."
        )

        st.markdown("#### Validación cruzada (5 folds, sobre el conjunto de entrenamiento)")
        nombres_metricas = {
            "accuracy": "Exactitud", "precision": "Precisión", "recall": "Recall",
            "f1": "F1-score", "roc_auc": "ROC-AUC",
        }
        filas = []
        for clave, nombre in nombres_metricas.items():
            c = metricas["cv"][clave]
            filas.append({
                "Métrica": nombre,
                "Validación (media)": round(c["val_media"], 3),
                "Desviación estándar": round(c["val_std"], 3),
                "Entrenamiento (media)": round(c["train_media"], 3),
            })
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.caption(
            "Una diferencia grande entre entrenamiento y validación indica sobreajuste; una "
            "desviación estándar alta indica inestabilidad entre particiones."
        )

    st.divider()
    st.markdown("### Evaluación gráfica")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### Matriz de confusión y curva ROC")
        mostrar_imagen("matriz_confusion_roc.png", "Matriz de confusión y curva ROC en el conjunto de prueba")
    with col_img2:
        st.markdown("#### Variables más importantes")
        mostrar_imagen("importancia_variables.png", "Importancia de las variables clínicas (Random Forest)")

    st.divider()
    n_limpio = (metricas or {}).get("n_limpio", 297)
    n_test = (metricas or {}).get("n_test", 60)
    st.markdown("### Limitaciones del modelo")
    st.info(
        f"""
* **Tamaño de la muestra:** se entrenó con {n_limpio} registros limpios del dataset de Cleveland y se
  evaluó con solo {n_test} pacientes. Con una prueba tan pequeña, cada paciente mal clasificado
  cambia las métricas en varios puntos porcentuales; por eso se reporta también la validación cruzada.
* **Origen de los datos:** los pacientes provienen de una sola institución y de una época específica,
  por lo que el modelo puede no generalizar a otras poblaciones.
* **Falsos negativos:** ningún modelo de este tipo es perfecto; un paciente enfermo puede recibir
  una probabilidad baja. Por eso el resultado no debe usarse para descartar una enfermedad.
* **Datos de entrada:** el modelo depende de la exactitud de los exámenes ingresados y no admite
  valores faltantes.
* **Variable objetivo binaria:** se agrupan los niveles de 1 a 4 de la variable original en
  "presencia de enfermedad", sin distinguir gravedad.
* **Uso restringido:** esta herramienta es un ejercicio académico y no está certificada para uso clínico.
"""
    )
