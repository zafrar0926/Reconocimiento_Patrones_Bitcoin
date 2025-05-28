import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import joblib
from pathlib import Path

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = ['modelos/xgboost', 'modelos/lightgbm', 'modelos/sarima']
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

def cargar_datos():
    """Cargar datos de entrenamiento"""
    print("Cargando datos...")
    df_train = pd.read_csv('datos_procesados/features/btc_features_train.csv')
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_train.set_index('timestamp', inplace=True)
    return df_train

def entrenar_xgboost(df_train, params, horizonte):
    """Entrenar modelo XGBoost con parámetros específicos"""
    print(f"\nEntrenando XGBoost para t+{horizonte}")
    target_col = f'target_t{horizonte}'
    feature_cols = [col for col in df_train.columns if col not in ['target_t1', 'target_t6', 'target_t12']]
    
    # Preparar datos
    X_train = df_train[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y_train = df_train[target_col]
    
    # Entrenar modelo
    modelo = XGBRegressor(**params)
    modelo.fit(X_train, y_train)
    
    # Guardar modelo
    joblib.dump(modelo, f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
    print(f"Modelo XGBoost t+{horizonte} guardado")

def entrenar_lightgbm(df_train, params, horizonte):
    """Entrenar modelo LightGBM con parámetros específicos"""
    print(f"\nEntrenando LightGBM para t+{horizonte}")
    target_col = f'target_t{horizonte}'
    feature_cols = [col for col in df_train.columns if col not in ['target_t1', 'target_t6', 'target_t12']]
    
    # Preparar datos
    X_train = df_train[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y_train = df_train[target_col]
    
    # Entrenar modelo
    modelo = LGBMRegressor(**params)
    modelo.fit(X_train, y_train)
    
    # Guardar modelo
    joblib.dump(modelo, f'modelos/lightgbm/modelo_lightgbm_t{horizonte}.joblib')
    print(f"Modelo LightGBM t+{horizonte} guardado")

def entrenar_sarima(df_train, params, horizonte):
    """Entrenar modelo SARIMA con parámetros específicos"""
    print(f"\nEntrenando SARIMA para t+{horizonte}")
    target_col = f'target_t{horizonte}'
    
    # Preparar serie temporal
    serie = df_train[target_col].copy()
    
    # Asegurar que el índice esté ordenado
    serie = serie.sort_index()
    
    # Calcular la diferencia de tiempo promedio entre observaciones
    tiempo_diff = serie.index.to_series().diff().median()
    if pd.Timedelta('1 hour') - pd.Timedelta('1 minute') <= tiempo_diff <= pd.Timedelta('1 hour') + pd.Timedelta('1 minute'):
        # Reindexar a frecuencia horaria si los datos son aproximadamente horarios
        idx_completo = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq='H')
        serie = serie.reindex(idx_completo)
        # Interpolar valores faltantes
        serie = serie.interpolate(method='time')
    else:
        print(f"ADVERTENCIA: La frecuencia de los datos no parece ser horaria (diferencia promedio: {tiempo_diff})")
    
    # Entrenar modelo
    modelo = SARIMAX(serie, 
                    order=params['order'],
                    seasonal_order=params['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    resultado = modelo.fit(disp=False)
    
    # Guardar modelo
    joblib.dump(resultado, f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
    print(f"Modelo SARIMA t+{horizonte} guardado")

def main():
    # Crear directorios
    crear_directorios()
    
    # Cargar hiperparámetros
    with open('mejores_hiperparametros.json', 'r') as f:
        params = json.load(f)
    
    # Cargar datos
    df_train = cargar_datos()
    
    # Para cada horizonte
    for horizonte in [1, 6, 12]:
        print(f"\nProcesando horizonte t+{horizonte}")
        
        # Entrenar XGBoost
        entrenar_xgboost(df_train, params['xgboost'][f't{horizonte}'], horizonte)
        
        # Entrenar LightGBM
        entrenar_lightgbm(df_train, params['lightgbm'][f't{horizonte}'], horizonte)
        
        # Entrenar SARIMA
        entrenar_sarima(df_train, params['sarima'][f't{horizonte}'], horizonte)
    
    print("\n¡Entrenamiento completado!")

if __name__ == "__main__":
    main() 