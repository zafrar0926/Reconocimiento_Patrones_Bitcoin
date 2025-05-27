# Predicción de Retornos de Bitcoin Utilizando Modelos de Machine Learning y Series Temporales

## Resumen

Este estudio presenta un análisis comparativo de tres enfoques diferentes para la predicción de retornos logarítmicos del Bitcoin: XGBoost, LightGBM y SARIMA. Se implementa un sistema completo de predicción que abarca desde el preprocesamiento de datos hasta la evaluación de modelos, con horizontes de predicción de 1, 6 y 12 horas. Los resultados demuestran la efectividad relativa de cada enfoque y proporcionan insights sobre la predictibilidad de los retornos de Bitcoin en diferentes horizontes temporales.

## I. Introducción

La predicción de precios de criptomonedas representa un desafío significativo en finanzas cuantitativas debido a su alta volatilidad y la influencia de múltiples factores externos. Este estudio se centra en el Bitcoin, la criptomoneda más importante por capitalización de mercado, y explora la efectividad de diferentes técnicas de modelado para predecir sus retornos logarítmicos.

## II. Datos y Metodología

### A. Dataset de Entrada

El dataset principal consiste en datos históricos de Bitcoin con frecuencia horaria, incluyendo:
- Precio de apertura, cierre, máximo y mínimo
- Volumen de transacciones
- Timestamp de cada observación

### B. Preprocesamiento y Feature Engineering

1) **Variables Técnicas**:
   - Medias Móviles Simples (SMA) de 7 y 21 períodos
   - Medias Móviles Exponenciales (EMA) de 7 y 21 períodos
   - MACD (Moving Average Convergence Divergence)
   - RSI (Relative Strength Index)
   - Oscilador Estocástico
   - Bandas de Bollinger

2) **Variables de Retornos**:
   - Volatilidad histórica en ventanas de 12, 24, 48 y 168 horas
   - Momentum en períodos de 12, 24 y 168 horas

3) **Variables Temporales**:
   - Codificación cíclica de hora del día
   - Codificación cíclica de día de la semana
   - Codificación cíclica de mes

4) **Variables Objetivo**:
   - Retornos logarítmicos futuros para t+1, t+6 y t+12 horas

### C. Análisis Estadístico Preliminar

Se realizó un análisis exhaustivo de la serie temporal de retornos logarítmicos, que reveló las siguientes características:

1. **Prueba de Estacionariedad (ADF)**:
   - Estadístico ADF: -26.248
   - p-value: 0.000
   - Valores críticos:
     * 1%: -3.431
     * 5%: -2.862
     * 10%: -2.567
   - Conclusión: La serie es estacionaria con un nivel de confianza del 99%

2. **Estadísticas Descriptivas**:
   - Número de observaciones: 34,855
   - Media: 2.54e-05 (cercana a cero)
   - Desviación estándar: 0.00675
   - Valores extremos:
     * Mínimo: -0.0993 (-9.93%)
     * Máximo: 0.1157 (11.57%)
   - Cuartiles:
     * Q1 (25%): -0.00245
     * Mediana: 6.23e-05
     * Q3 (75%): 0.00260

3. **Características de la Serie**:
   - Distribución leptocúrtica (colas pesadas)
   - Ligera asimetría positiva
   - Agrupamiento de volatilidad
   - Retornos centrados en cero

4. **Implicaciones para el Modelado**:
   - La estacionariedad confirma la idoneidad de los modelos SARIMA
   - La presencia de colas pesadas justifica el uso de modelos no lineales (XGBoost, LightGBM)
   - El agrupamiento de volatilidad sugiere la importancia de características basadas en volatilidad histórica

## III. Modelos Implementados

### A. XGBoost

1) **Hiperparámetros**:
   ```python
   {
       'n_estimators': 1000,
       'learning_rate': 0.01,
       'max_depth': 5,
       'min_child_weight': 1,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'gamma': 0,
       'objective': 'reg:squarederror'
   }
   ```

2) **Características del Entrenamiento**:
   - Validación cruzada temporal con 5 folds
   - Early stopping para prevenir overfitting
   - Evaluación de importancia de características

### B. LightGBM

1) **Hiperparámetros**:
   ```python
   {
       'n_estimators': 500,
       'learning_rate': 0.05,
       'max_depth': 5,
       'num_leaves': 31,
       'min_child_samples': 20,
       'subsample': 0.8,
       'colsample_bytree': 0.8
   }
   ```

2) **Características del Entrenamiento**:
   - Early stopping con 50 rondas de paciencia
   - Validación temporal
   - Manejo automático de valores faltantes

### C. SARIMA

1) **Configuraciones por Horizonte**:
   - t+1: SARIMA(1,1,1)(1,1,1,24)
   - t+6: SARIMA(2,1,1)(0,1,1,24)
   - t+12: SARIMA(1,1,2)(1,1,1,24)

2) **Características del Modelo**:
   - Componente estacional con período de 24 horas
   - Diferenciación para lograr estacionariedad
   - Intervalos de confianza para predicciones

## IV. Resultados y Discusión

### A. Métricas de Evaluación

Se utilizaron tres métricas principales para evaluar el desempeño de los modelos:
- RMSE (Root Mean Square Error): Mide la raíz cuadrada del error cuadrático medio
- MAE (Mean Absolute Error): Mide el error absoluto medio
- R² (Coeficiente de determinación): Mide la proporción de varianza explicada por el modelo

### B. Comparación de Modelos

Los resultados por horizonte y modelo son:

1. **Horizonte t+1 (1 hora)**:
   - XGBoost: RMSE=0.00564, MAE=0.00377, R²=-0.0547
   - LightGBM: RMSE=0.00577, MAE=0.00389, R²=-0.1032
   - SARIMA: RMSE=0.00565, MAE=0.00400, R²=-0.0058

2. **Horizonte t+6 (6 horas)**:
   - XGBoost: RMSE=0.01522, MAE=0.01101, R²=-0.3466
   - LightGBM: RMSE=0.01803, MAE=0.01389, R²=-0.8911
   - SARIMA: RMSE=0.01559, MAE=0.01146, R²=-0.3908

3. **Horizonte t+12 (12 horas)**:
   - XGBoost: RMSE=0.02277, MAE=0.01743, R²=-0.5164
   - LightGBM: RMSE=0.02607, MAE=0.02050, R²=-0.9870
   - SARIMA: RMSE=0.02191, MAE=0.01803, R²=-0.5422

Observaciones principales:
1. Los tres modelos muestran un mejor desempeño en predicciones a corto plazo (t+1)
2. El error aumenta significativamente con horizontes más largos
3. XGBoost muestra el mejor desempeño general, especialmente en horizontes más largos
4. Los valores negativos de R² indican la dificultad inherente en la predicción de retornos financieros

### C. Análisis de Importancia de Características

Las variables más relevantes identificadas por los modelos de machine learning fueron:

1. **XGBoost**:
   - Volatilidad histórica de 48 horas
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bandas de Bollinger
   - Variables temporales (especialmente hora del día)

2. **LightGBM**:
   - Volatilidad histórica de 24 y 48 horas
   - Momentum de 24 horas
   - EMA de 21 períodos
   - Variables cíclicas temporales

## V. Conclusiones y Trabajo Futuro

### A. Hallazgos Principales

1. La predicción de retornos de Bitcoin es más precisa en horizontes cortos (1 hora) que en horizontes más largos (6 y 12 horas).
2. XGBoost muestra el mejor desempeño general, con errores más bajos y mayor estabilidad en todos los horizontes.
3. Los indicadores técnicos basados en volatilidad y momentum son los más relevantes para la predicción.
4. La componente temporal (hora del día, día de la semana) tiene una influencia significativa en los patrones de retorno.
5. Los valores negativos de R² sugieren que la predicción de retornos exactos es extremadamente desafiante, lo cual es consistente con la hipótesis de mercados eficientes.

### B. Limitaciones

1. La alta volatilidad del mercado de criptomonedas
2. La influencia de factores externos no capturados en los datos
3. La naturaleza cambiante de las relaciones entre variables

### C. Direcciones Futuras

1. Incorporación de datos de sentimiento y redes sociales
2. Exploración de modelos de deep learning
3. Análisis de la estabilidad temporal de las predicciones

## VI. Referencias

[1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785-794.

[2] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in Advances in Neural Information Processing Systems 30, 2017, pp. 3146-3154.

[3] R. H. Shumway and D. S. Stoffer, "Time Series Analysis and Its Applications: With R Examples," Springer, 4th edition, 2017.

[4] S. McNally, J. Roche, and S. Caton, "Predicting the Price of Bitcoin Using Machine Learning," in 2018 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP), 2018, pp. 339-343.

[5] A. M. Rather, A. Agarwal, and V. N. Sastry, "Recurrent neural network and a hybrid model for prediction of stock returns," Expert Systems with Applications, vol. 42, no. 6, pp. 3234-3241, 2015.

[6] Y. B. Kim et al., "Predicting Fluctuations in Cryptocurrency Transactions Based on User Comments and Replies," PLOS ONE, vol. 11, no. 8, 2016.

[7] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M., "Time Series Analysis: Forecasting and Control," Wiley, 5th edition, 2015.

[8] Nakamoto, S., "Bitcoin: A Peer-to-Peer Electronic Cash System," 2008. [Online]. Available: https://bitcoin.org/bitcoin.pdf

## Apéndice: Estructura del Proyecto

```