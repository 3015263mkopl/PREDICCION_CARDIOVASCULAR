# Sistema de Predicción Cardiovascular

Aplicación web en Streamlit que estima la probabilidad de presencia de enfermedad cardiovascular a partir de variables clínicas, usando un modelo Random Forest entrenado con el dataset Heart Disease de Cleveland (UCI).

**Aviso:** es un prototipo académico. Las predicciones no sustituyen el diagnóstico ni el consejo de un profesional de la salud.

- Aplicación pública: ENLACE_DE_STREAMLIT_COMMUNITY_CLOUD
- Curso: Minería de Datos, Actividad 1 del tercer corte

## Integrantes

- Héctor Cervantes Echeverria
- Juan Guillermo Sarmiento Garrido
- Santiago Valle Palacio
- Diego Mendoza Gutiérrez

## Datos

Fuente: UCI Machine Learning Repository, Heart Disease (subconjunto de Cleveland). https://archive.ics.uci.edu/dataset/45/heart+disease

Se trabajó con 303 registros originales; tras eliminar valores faltantes y duplicados quedaron 297. La variable objetivo original (0 a 4) se binarizó: 0 = sin enfermedad, 1 = con enfermedad (valores 1 a 4).

## Modelo

- Pipeline completo guardado con `joblib` (`modelo_pipeline.pkl`): `StandardScaler` para variables continuas, `OneHotEncoder` para categóricas y `RandomForestClassifier`.
- División estratificada 80 % entrenamiento y 20 % prueba (`random_state=42`).
- Selección del modelo final por recall en validación cruzada de 5 folds sobre el entrenamiento (no sobre la prueba), comparando Random Forest por defecto contra Random Forest optimizado con `GridSearchCV`.
- La aplicación carga el pipeline guardado y no entrena nada al ejecutarse.

### Resultados

Pegar aquí el contenido del archivo `resultados.md` que genera `entrenar_modelo.py`.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación de Streamlit |
| `requirements.txt` | Dependencias (generado por el script con la versión exacta de scikit-learn) |
| `modelo_pipeline.pkl` | Pipeline entrenado |
| `metricas.json` | Métricas reales de prueba y validación cruzada que muestra la app |
| `ejemplos_test.json` | Registros conocidos del conjunto de prueba para verificar la app |
| `matriz_confusion_roc.png`, `importancia_variables.png`, `eda_figura.png` | Figuras mostradas en la app |
| `entrenar_modelo.py` | Script que reentrena y genera todos los archivos anteriores |
| Notebooks | Notebooks de los cortes 1 y 2 |

## Cómo reproducir

1. Ejecutar `entrenar_modelo.py` en Google Colab. Descarga `archivos_app.zip` con el modelo, las métricas, las figuras y el `requirements.txt`.
2. Subir esos archivos al repositorio junto con `app.py`.
3. Localmente: `pip install -r requirements.txt` y luego `streamlit run app.py`.
4. Despliegue: en Streamlit Community Cloud, crear una app apuntando a este repositorio y a `app.py`. Si la instalación falla, elegir en Advanced settings la misma versión de Python de Colab.

## Pruebas realizadas

- Caso normal: paciente de valores típicos, la app devuelve predicción y probabilidad.
- Entrada inválida: valores fuera de rango (por ejemplo edad 150) son rechazados por el formulario y por la validación del servidor; valores atípicos pero posibles (por ejemplo colesterol mayor a 400) generan una advertencia.
- Registro conocido: en la sección "Verificación con un registro conocido del conjunto de prueba" la app calcula la predicción y la compara con la del notebook; deben coincidir.

## Evaluación del uso de inteligencia artificial

- **Herramienta utilizada:** Claude (Anthropic).
- **Propósito:** revisar si la aplicación cumplía la rúbrica de la actividad, detectar errores y generar el script de reentrenamiento, la versión corregida de `app.py` y este README.
- **Prompt principal:** se compartió el enunciado de la actividad y el código de `app.py` preguntando si cumplía con todo; después se aportaron los notebooks de los cortes 1 y 2 para contrastar el código con los resultados reales.
- **Cómo se verificó la respuesta:** se comparó cada requisito de la rúbrica con el código; se contrastaron las métricas de la app con la tabla del notebook; se ejecutó el script de entrenamiento en Colab y se comprobó que la predicción de la app coincide con la del notebook para registros del conjunto de prueba.
- **Problemas encontrados:** faltaban las visualizaciones del EDA; las dos imágenes del modelo estaban intercambiadas; las métricas estaban escritas a mano y correspondían al Random Forest sin optimizar, mientras las gráficas decían "optimizado"; el GridSearch del notebook había sido interrumpido y nunca se guardaba el modelo con `joblib`; la explicación de variables importantes no estaba verificada; no se reportaba validación cruzada.
- **Ajustes realizados:** se agregó la pestaña de EDA, se corrigieron los nombres de las imágenes, las métricas se leen de `metricas.json` generado por el entrenamiento, se agregó validación de entradas con advertencias, se añadió la verificación contra el notebook y se fijó la versión de scikit-learn.
- Todos los integrantes revisaron y comprenden el código entregado.
