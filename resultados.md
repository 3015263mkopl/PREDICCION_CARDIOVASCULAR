Modelo final: Random Forest optimizado (GridSearchCV)

| Metrica | Prueba (n=60) | Validacion cruzada (media +/- desv.) |
|---|---|---|
| Exactitud | 0.850 | 0.806 +/- 0.066 |
| Precision | 0.880 | 0.833 +/- 0.113 |
| Recall | 0.786 | 0.751 +/- 0.114 |
| F1-score | 0.830 | 0.780 +/- 0.068 |
| ROC-AUC | 0.920 | 0.891 +/- 0.068 |

Matriz de confusion en prueba: VN=29, FP=3, FN=6, VP=22. Especificidad: 0.906.
Mejores parametros: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 200}
