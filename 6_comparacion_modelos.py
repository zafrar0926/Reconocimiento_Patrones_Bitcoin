import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = ['graficos_comparacion', 'resultados/comparacion']
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

def cargar_resultados(archivos):
    """Cargar resultados de los diferentes modelos"""
    resultados = {}
    for nombre, archivo in archivos.items():
        try:
            df = pd.read_csv(archivo)
            
            # Manejar diferentes formatos de CSV
            if df.columns[0] in ['Unnamed: 0', 'target', 'index']:
                df.set_index(df.columns[0], inplace=True)
            
            # Si el formato es el de SARIMA (métricas en filas)
            if 't+1' in df.columns:
                # Crear nuevo DataFrame con el formato correcto
                new_df = pd.DataFrame(index=[f'target_t{idx}' for idx in [1, 6, 12]])
                new_df['rmse'] = df.loc['rmse', ['t+1', 't+6', 't+12']].values
                new_df['mae'] = df.loc['mae', ['t+1', 't+6', 't+12']].values
                new_df['r2'] = df.loc['r2', ['t+1', 't+6', 't+12']].values
                df = new_df
            
            resultados[nombre] = df
            
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {archivo}")
        except Exception as e:
            print(f"Error al procesar {archivo}: {str(e)}")
    
    return resultados

def plot_comparacion_metricas(resultados, metrica, titulo, guardar_como=None):
    """Generar gráfico comparativo de métricas"""
    plt.figure(figsize=(12, 6))
    data = []
    modelos = []
    horizontes = []
    valores = []
    
    for modelo, df in resultados.items():
        for horizonte in df.index:
            modelos.append(modelo)
            horizontes.append(horizonte)
            valores.append(df.loc[horizonte, metrica])
    
    df_plot = pd.DataFrame({
        'Modelo': modelos,
        'Horizonte': horizontes,
        'Valor': valores
    })
    
    sns.barplot(data=df_plot, x='Horizonte', y='Valor', hue='Modelo')
    plt.title(f'Comparación de {titulo} por Modelo y Horizonte')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como)
    plt.close()

def generar_predicciones_futuras(modelo_xgb, modelo_lgb, modelo_sarima, df_reciente, feature_cols, horizonte):
    """Generar predicciones futuras para todos los modelos"""
    try:
        # Preparar datos para modelos de ML
        X_reciente = df_reciente[feature_cols].iloc[-1:].copy()
        
        # Rellenar NaN si existen
        if X_reciente.isna().any().any():
            X_reciente = X_reciente.fillna(method='ffill').fillna(method='bfill')
        
        # Predicciones XGBoost
        pred_xgb = modelo_xgb.predict(X_reciente)[0]
        
        # Predicciones LightGBM
        pred_lgb = modelo_lgb.predict(X_reciente)[0]
        
        # Predicciones SARIMA
        pred_sarima = modelo_sarima.forecast(steps=1)[0]
        sarima_ci = pd.DataFrame(
            modelo_sarima.get_forecast(steps=1).conf_int(),
            columns=['lower', 'upper']
        )
        
        return pred_xgb, pred_lgb, pred_sarima, sarima_ci
    
    except Exception as e:
        print(f"Error al generar predicciones: {str(e)}")
        return None, None, None, None

def main():
    # Crear directorios necesarios
    crear_directorios()
    
    # Definir archivos de resultados
    archivos_resultados = {
        'XGBoost': 'resultados/xgboost/resultados_xgboost.csv',
        'LightGBM': 'resultados/lightgbm/resultados_lightgbm.csv',
        'SARIMA': 'resultados/sarima/resultados_sarima.csv'
    }
    
    # Cargar resultados
    print("Cargando resultados de los modelos...")
    resultados = cargar_resultados(archivos_resultados)
    
    if not resultados:
        print("No se encontraron resultados para comparar.")
        return
    
    # Generar gráficos comparativos
    print("\nGenerando gráficos comparativos...")
    metricas = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²'
    }
    
    for metrica, titulo in metricas.items():
        plot_comparacion_metricas(
            resultados,
            metrica,
            titulo,
            f'graficos_comparacion/comparacion_{metrica}.png'
        )
    
    # Generar tabla comparativa
    print("\nGenerando tabla comparativa...")
    tabla_comparativa = pd.DataFrame()
    
    for modelo, df in resultados.items():
        for metrica in metricas.keys():
            for horizonte in df.index:
                tabla_comparativa.loc[f"{modelo}_{horizonte}", metrica] = \
                    df.loc[horizonte, metrica]
    
    tabla_comparativa.to_csv('resultados/comparacion/comparacion_modelos_resumen.csv')
    
    # Generar predicciones futuras
    print("\nGenerando predicciones futuras...")
    try:
        # Cargar datos más recientes
        df_reciente = pd.read_csv('datos_procesados/features/btc_features_test.csv')
        df_reciente['timestamp'] = pd.to_datetime(df_reciente['timestamp'])
        df_reciente.set_index('timestamp', inplace=True)
        
        # Preparar datos
        target_cols = ['target_t1', 'target_t6', 'target_t12']
        feature_cols = [col for col in df_reciente.columns if col not in target_cols]
        
        # Generar predicciones para cada horizonte
        for horizonte in [1, 6, 12]:
            print(f"\nPredicciones para horizonte t+{horizonte}")
            try:
                # Cargar modelos
                modelo_xgb = joblib.load(f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
                modelo_lgb = joblib.load(f'modelos/lightgbm/modelo_lightgbm_t{horizonte}.joblib')
                modelo_sarima = joblib.load(f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
                
                # Generar predicciones
                pred_xgb, pred_lgb, pred_sarima, sarima_ci = generar_predicciones_futuras(
                    modelo_xgb, modelo_lgb, modelo_sarima,
                    df_reciente, feature_cols, horizonte
                )
                
                if pred_xgb is not None:
                    print(f"XGBoost: {pred_xgb:.6f}")
                    print(f"LightGBM: {pred_lgb:.6f}")
                    print(f"SARIMA: {pred_sarima:.6f} [{sarima_ci['lower'].iloc[0]:.6f}, {sarima_ci['upper'].iloc[0]:.6f}]")
                
            except Exception as e:
                print(f"Error al generar predicciones para t+{horizonte}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error al generar predicciones futuras: {str(e)}")
    
    print("\n¡Proceso completado!")
    print("Los resultados han sido guardados en:")
    print("- graficos_comparacion/")
    print("- resultados/comparacion/comparacion_modelos_resumen.csv")

if __name__ == "__main__":
    main() 