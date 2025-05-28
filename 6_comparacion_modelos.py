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

def plot_predicciones_vs_real(df_test, predicciones, horizonte, guardar_como=None):
    """Generar gráfico de predicciones vs valores reales"""
    plt.figure(figsize=(15, 7))
    
    # Valores reales
    plt.plot(df_test.index, df_test[f'target_t{horizonte}'], 
             color='black', label='Valores Reales', linewidth=2)
    
    # Predicciones de cada modelo
    colores = {'XGBoost': 'blue', 'LightGBM': 'green', 'SARIMA': 'red'}
    for modelo, pred in predicciones.items():
        plt.plot(df_test.index, pred, 
                color=colores[modelo], label=f'Predicciones {modelo}', 
                alpha=0.7, linewidth=1.5)
    
    plt.title(f'Predicciones vs Valores Reales (Horizonte t+{horizonte})')
    plt.xlabel('Fecha')
    plt.ylabel('Retorno Logarítmico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribucion_errores(errores, horizonte, guardar_como=None):
    """Generar gráfico de distribución de errores"""
    plt.figure(figsize=(12, 6))
    
    for modelo, error in errores.items():
        sns.kdeplot(data=error, label=modelo)
    
    plt.title(f'Distribución de Errores de Predicción (Horizonte t+{horizonte})')
    plt.xlabel('Error')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Crear directorios necesarios
    crear_directorios()
    
    # Cargar datos de test
    print("Cargando datos...")
    df_test = pd.read_csv('datos_procesados/features/btc_features_test.csv')
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    df_test.set_index('timestamp', inplace=True)
    
    # Preparar features
    target_cols = ['target_t1', 'target_t6', 'target_t12']
    feature_cols = [col for col in df_test.columns if col not in target_cols]
    
    # Para cada horizonte
    for horizonte in [1, 6, 12]:
        print(f"\nProcesando horizonte t+{horizonte}")
        predicciones = {}
        errores = {}
        
        # Cargar y predecir con XGBoost
        try:
            modelo_xgb = joblib.load(f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
            pred_xgb = modelo_xgb.predict(df_test[feature_cols])
            predicciones['XGBoost'] = pred_xgb
            errores['XGBoost'] = pred_xgb - df_test[f'target_t{horizonte}']
        except Exception as e:
            print(f"Error al cargar/predecir XGBoost: {str(e)}")
        
        # Cargar y predecir con LightGBM
        try:
            modelo_lgb = joblib.load(f'modelos/lightgbm/modelo_lightgbm_t{horizonte}.joblib')
            pred_lgb = modelo_lgb.predict(df_test[feature_cols])
            predicciones['LightGBM'] = pred_lgb
            errores['LightGBM'] = pred_lgb - df_test[f'target_t{horizonte}']
        except Exception as e:
            print(f"Error al cargar/predecir LightGBM: {str(e)}")
        
        # Cargar y predecir con SARIMA
        try:
            modelo_sarima = joblib.load(f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
            pred_sarima = modelo_sarima.get_forecast(steps=len(df_test)).predicted_mean
            predicciones['SARIMA'] = pred_sarima
            errores['SARIMA'] = pred_sarima - df_test[f'target_t{horizonte}']
        except Exception as e:
            print(f"Error al cargar/predecir SARIMA: {str(e)}")
        
        if predicciones:
            # Generar gráfico de predicciones vs real
            plot_predicciones_vs_real(
                df_test,
                predicciones,
                horizonte,
                f'graficos_comparacion/predicciones_vs_real_t{horizonte}.png'
            )
            
            # Generar gráfico de distribución de errores
            plot_distribucion_errores(
                errores,
                horizonte,
                f'graficos_comparacion/distribucion_errores_t{horizonte}.png'
            )
    
    print("\n¡Proceso completado!")
    print("Los resultados han sido guardados en:")
    print("- graficos_comparacion/")

if __name__ == "__main__":
    main() 