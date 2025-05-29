import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime, timedelta

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = ['graficos_precios']
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

def retorno_a_precio(precio_inicial, retornos):
    """Convierte retornos logarítmicos a precios"""
    precios = [precio_inicial]
    for retorno in retornos:
        precio_siguiente = precios[-1] * np.exp(retorno)
        precios.append(precio_siguiente)
    return precios[1:]  # Excluimos el precio inicial que ya teníamos

def plot_predicciones_precio(df_historico, predicciones, horizonte, precio_inicial, guardar_como=None):
    """Generar gráfico de predicciones de precios vs precios reales"""
    plt.figure(figsize=(15, 7))
    
    # Convertir retornos a precios
    precios_reales = retorno_a_precio(precio_inicial, df_historico[f'target_t{horizonte}'])
    
    # Graficar precios históricos (primeros 2000 registros)
    plt.plot(df_historico.index[:2000], precios_reales[:2000], 
             color='black', label='Precios Reales', linewidth=2)
    
    # Para los últimos 500 registros
    plt.plot(df_historico.index[2000:], precios_reales[2000:], 
             color='black', linewidth=2)
    
    # Predicciones de cada modelo (solo últimos 500 registros)
    colores = {'XGBoost': 'blue', 'LightGBM': 'green', 'SARIMA': 'red'}
    for modelo, retornos_pred in predicciones.items():
        precios_pred = retorno_a_precio(precios_reales[1999], retornos_pred)  # Usar último precio real como inicial
        plt.plot(df_historico.index[2000:], precios_pred, 
                color=colores[modelo], label=f'Predicciones {modelo}', 
                alpha=0.7, linewidth=1.5)
    
    plt.title(f'Predicción de Precios de Bitcoin (Horizonte t+{horizonte})')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Crear directorios necesarios
    crear_directorios()
    
    # Cargar datos históricos
    print("Cargando datos...")
    df_historico = pd.read_csv('datos_procesados/features/btc_features_test.csv')
    df_historico['timestamp'] = pd.to_datetime(df_historico['timestamp'])
    df_historico.set_index('timestamp', inplace=True)
    
    # Usar los últimos 2500 registros
    df_historico = df_historico.tail(2500)
    print(f"\nUsando los últimos {len(df_historico)} registros")
    print(f"Rango de fechas: {df_historico.index[0]} a {df_historico.index[-1]}")
    
    # Preparar features
    target_cols = ['target_t1', 'target_t6', 'target_t12']
    feature_cols = [col for col in df_historico.columns if col not in target_cols]
    
    # Para cada horizonte
    for horizonte in [1, 6, 12]:
        print(f"\nProcesando horizonte t+{horizonte}")
        predicciones = {}
        
        # Últimos 500 registros para predicciones
        df_pred = df_historico.tail(500)
        X_pred = df_pred[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Cargar y predecir con XGBoost
        try:
            print(f"Cargando modelo XGBoost t+{horizonte}...")
            modelo_xgb = joblib.load(f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
            predicciones['XGBoost'] = modelo_xgb.predict(X_pred)
            print("XGBoost: OK")
        except Exception as e:
            print(f"Error al cargar/predecir XGBoost t+{horizonte}: {e}")
        
        # Cargar y predecir con LightGBM
        try:
            print(f"Cargando modelo LightGBM t+{horizonte}...")
            modelo_lgb = joblib.load(f'modelos/lightgbm/modelo_lightgbm_t{horizonte}.joblib')
            predicciones['LightGBM'] = modelo_lgb.predict(X_pred)
            print("LightGBM: OK")
        except Exception as e:
            print(f"Error al cargar/predecir LightGBM t+{horizonte}: {e}")
        
        # Cargar y predecir con SARIMA
        try:
            print(f"Cargando modelo SARIMA t+{horizonte}...")
            modelo_sarima = joblib.load(f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
            serie_temporal = df_pred[f'target_t{horizonte}'].copy()
            predicciones_sarima = modelo_sarima.get_forecast(steps=len(serie_temporal))
            predicciones['SARIMA'] = predicciones_sarima.predicted_mean.values
            print("SARIMA: OK")
        except Exception as e:
            print(f"Error al cargar/predecir SARIMA t+{horizonte}: {e}")
        
        # Obtener precio inicial (último precio real antes de las predicciones)
        precio_inicial = df_historico['high'].iloc[1999]  # Último precio antes de las predicciones
        
        # Generar gráfico
        print("Generando gráfico de precios...")
        plot_predicciones_precio(df_historico, predicciones, horizonte, precio_inicial,
                               f'graficos_precios/prediccion_precios_t{horizonte}.png')
    
    print("\n¡Proceso completado!")
    print("Los gráficos han sido guardados en:")
    print("- graficos_precios/")

if __name__ == "__main__":
    main() 