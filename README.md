# Sistema de Predicción Cardiovascular

Aplicación web en Streamlit que estima la probabilidad de presencia de enfermedad cardiovascular a partir de variables clínicas, usando un modelo Random Forest entrenado con el dataset Heart Disease de Cleveland (UCI).

**Aviso:** es un prototipo académico. Las predicciones no sustituyen el diagnóstico ni el consejo de un profesional de la salud.

- Aplicación pública: https://prediccioncardiovascular-version-1.streamlit.app/
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

## Resultados

Modelo final: Random Forest optimizado (GridSearchCV), seleccionado por recall en validación cruzada.
Mejores parámetros: `criterion=gini`, `max_depth=5`, `min_samples_split=10`, `n_estimators=200`.

| Métrica | Prueba (n=60) | Validación cruzada (media +/- desv.) |
|---|---|---|
| Exactitud | 0.850 | 0.806 +/- 0.066 |
| Precisión | 0.880 | 0.833 +/- 0.113 |
| Recall | 0.786 | 0.751 +/- 0.114 |
| F1-score | 0.830 | 0.780 +/- 0.068 |
| ROC-AUC | 0.920 | 0.891 +/- 0.068 |

Matriz de confusión en prueba: 29 verdaderos negativos, 3 falsos positivos, 6 falsos negativos y 22 verdaderos positivos. Especificidad: 0.906.

### Revisión de calidad del modelo

La exactitud (0.85) supera el umbral de revisión de 0.70. Aun así se revisó el desempeño por clase y la estabilidad:

- El modelo no detecta 6 de los 28 pacientes enfermos del conjunto de prueba (21 %), lo cual es relevante en un contexto médico.
- El modelo rinde mejor en entrenamiento que en validación cruzada (exactitud 0.923 frente a 0.806, recall 0.890 frente a 0.751 y ROC-AUC 0.978 frente a 0.891), lo que indica un sobreajuste moderado. El resultado en prueba (exactitud 0.850) queda algo por encima de la validación cruzada, por lo que el rendimiento esperado en pacientes nuevos es más cercano a 0.80 que a 0.85.
- El recall varía bastante entre particiones (desviación de 0.114) y el conjunto de prueba es pequeño (60 pacientes): un solo paciente cambia el recall en 3.6 puntos porcentuales.

### Cambios respecto al segundo corte

En el segundo corte el GridSearch no se completó y el modelo no se guardaba. Ahora el GridSearch termina, el modelo final se elige solo con validación cruzada (sin usar el conjunto de prueba) y el pipeline completo se guarda con `joblib`.

| Métrica (prueba) | Segundo corte (Random Forest por defecto) | Tercer corte (Random Forest optimizado) |
|---|---|---|
| Exactitud | 85.0 % | 85.0 % |
| Recall | 82.1 % | 78.6 % |
| Precisión | 85.1 % | 88.0 % |
| F1-score | 83.6 % | 83.0 % |

La diferencia en recall equivale a un solo paciente del conjunto de prueba (23 frente a 22 enfermos detectados), por lo que no se considera una mejora significativa: la optimización de hiperparámetros no aportó una ganancia clara en este conjunto de datos.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación de Streamlit |
| `requirements.txt` | Dependencias (con la versión exacta de scikit-learn usada al entrenar) |
| `modelo_pipeline.pkl` | Pipeline entrenado |
| `metricas.json` | Métricas reales de prueba y validación cruzada que muestra la app |
| `ejemplos_test.json` | Registros conocidos del conjunto de prueba para verificar la app |
| `matriz_confusion_roc.png`, `importancia_variables.png`, `eda_figura.png` | Figuras mostradas en la app |
| `resultados.md` | Tabla de resultados generada por el entrenamiento |
| `entrenar_modelo.py` | Script que reentrena y genera todos los archivos anteriores |
| Notebook | Notebook actualizado con el reentrenamiento y sus salidas |

## Cómo reproducir

1. Ejecutar `entrenar_modelo.py` en Google Colab. Descarga `archivos_app.zip` con el modelo, las métricas, las figuras y el `requirements.txt`.
2. Subir esos archivos al repositorio junto con `app.py`.
3. Localmente: `pip install -r requirements.txt` y luego `streamlit run app.py`.
4. Despliegue: en Streamlit Community Cloud, crear una app apuntando a este repositorio y a `app.py`. Si la instalación falla, elegir en Advanced settings la misma versión de Python de Colab.

## Pruebas realizadas

- **Caso normal:** paciente de valores típicos; la app devuelve predicción y probabilidad.
- **Entrada inválida:** valores fuera de rango (por ejemplo edad 150) son rechazados por el formulario y por la validación del servidor; valores atípicos pero posibles (por ejemplo colesterol mayor a 400) generan una advertencia.
- **Registro conocido:** en la sección "Verificación con un registro conocido del conjunto de prueba" la app calcula la predicción con el modelo guardado y la compara con la obtenida en el notebook para el mismo registro; ambas deben coincidir.

## Evaluación del uso de inteligencia artificial

- **Herramienta utilizada:** Claude (Anthropic).
- **Propósito:** revisar si la aplicación cumplía la rúbrica de la actividad, detectar errores y generar el script de reentrenamiento, la versión corregida de `app.py` y este README.
- **Prompt principal:** se compartió el enunciado de la actividad y el código de `app.py` preguntando si cumplía con todo; después se aportaron los notebooks de los cortes 1 y 2 para contrastar el código con los resultados reales.
- **Cómo se verificó la respuesta:** se comparó cada requisito de la rúbrica con el código; se contrastaron las métricas de la app con la tabla del notebook; se ejecutó el script de entrenamiento en Colab, se comprobó que las métricas reportadas sean consistentes entre sí (por ejemplo, que la matriz de confusión reproduzca la exactitud, precisión y recall) y se comparó la predicción de la app con la del notebook para registros del conjunto de prueba.
- **Problemas encontrados:** faltaban las visualizaciones del EDA; las dos imágenes del modelo estaban intercambiadas; las métricas estaban escritas a mano y correspondían al Random Forest sin optimizar, mientras las gráficas decían "optimizado"; el GridSearch del notebook había sido interrumpido y nunca se guardaba el modelo con `joblib`; la explicación de variables importantes no estaba verificada; no se reportaba validación cruzada.
- **Ajustes realizados:** se agregó la pestaña de EDA, se corrigieron los nombres de las imágenes, las métricas se leen de `metricas.json` generado por el entrenamiento, se agregó validación de entradas con advertencias, se añadió la verificación contra el notebook, se eligió el modelo final por validación cruzada y se fijó la versión de scikit-learn.
