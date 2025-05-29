import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta
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

def generar_tabla_predicciones(df_test, predicciones, horizonte=1, n_muestras=10):
    """Generar tabla comparativa de las últimas n predicciones"""
    # Crear DataFrame con timestamp y valor real
    tabla = pd.DataFrame({
        'Fecha': df_test.index[-n_muestras:],
        'Real': df_test[f'target_t{horizonte}'].iloc[-n_muestras:]
    })
    
    # Agregar predicciones de cada modelo
    for modelo, pred in predicciones.items():
        tabla[modelo] = pred[-n_muestras:]
    
    # Formatear la tabla
    tabla['Fecha'] = tabla['Fecha'].dt.strftime('%Y-%m-%d %H:%M')
    for col in tabla.columns[1:]:  # Todas las columnas excepto Fecha
        tabla[col] = tabla[col].map('{:.6f}'.format)
    
    # Guardar tabla en formato markdown
    with open(f'resultados/comparacion/tabla_predicciones_t{horizonte}.md', 'w') as f:
        f.write(f"### Comparación de Predicciones para Horizonte t+{horizonte}\n\n")
        f.write(tabla.to_markdown(index=False))

def plot_metricas_comparativas(metricas_por_horizonte):
    """Generar gráficos comparativos de métricas para todos los horizontes"""
    # Preparar datos para plotting
    horizontes = [1, 6, 12]
    modelos = list(metricas_por_horizonte[1].keys())  # Obtener nombres de modelos
    
    # Para cada métrica (RMSE, MAE, R2)
    for metrica in ['RMSE', 'MAE', 'R2']:
        plt.figure(figsize=(12, 6))
        
        # Preparar datos para el gráfico
        valores = {modelo: [] for modelo in modelos}
        for horizonte in horizontes:
            for modelo in modelos:
                valores[modelo].append(metricas_por_horizonte[horizonte][modelo][metrica])
        
        # Crear el gráfico de barras agrupadas
        x = np.arange(len(horizontes))
        width = 0.25  # Ancho de las barras
        
        # Plotear barras para cada modelo
        for i, (modelo, vals) in enumerate(valores.items()):
            plt.bar(x + i*width - width, vals, width, label=modelo)
        
        plt.xlabel('Horizonte de Predicción')
        plt.ylabel(metrica)
        plt.title(f'Comparación de {metrica} por Modelo y Horizonte')
        plt.xticks(x, [f't+{h}' for h in horizontes])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar el gráfico
        plt.savefig(f'graficos_comparacion/comparacion_{metrica.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Crear directorios necesarios
    crear_directorios()
    
    # Cargar datos de test
    print("Cargando datos...")
    df_test = pd.read_csv('datos_procesados/features/btc_features_test.csv')
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    df_test.set_index('timestamp', inplace=True)
    
    # Usar solo los últimos 500 datos para comparación justa con SARIMA
    df_test = df_test.tail(500)
    print(f"\nUsando los últimos {len(df_test)} datos para comparación")
    print(f"Rango de fechas: {df_test.index[0]} a {df_test.index[-1]}")
    
    # Preparar features
    target_cols = ['target_t1', 'target_t6', 'target_t12']
    feature_cols = [col for col in df_test.columns if col not in target_cols]
    
    print("\nColumnas de features disponibles:")
    print(feature_cols)
    print("\nPrimeras 5 filas de features:")
    print(df_test[feature_cols].head())
    
    # Almacenar métricas de todos los horizontes
    metricas_por_horizonte = {}
    
    # Para cada horizonte
    for horizonte in [1, 6, 12]:
        print(f"\nProcesando horizonte t+{horizonte}")
        predicciones = {}
        errores = {}
        
        # Cargar y predecir con XGBoost
        try:
            print(f"Cargando modelo XGBoost t+{horizonte}...")
            modelo_xgb = joblib.load(f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
            print("Modelo XGBoost cargado. Feature importances:")
            print(pd.Series(modelo_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head())
            
            X_test = df_test[feature_cols].fillna(method='ffill').fillna(method='bfill')
            print("\nEstadísticas de los features procesados:")
            print(X_test.describe())
            
            predicciones['XGBoost'] = modelo_xgb.predict(X_test)
            print("\nEstadísticas de las predicciones XGBoost:")
            print(pd.Series(predicciones['XGBoost']).describe())
            
            errores['XGBoost'] = predicciones['XGBoost'] - df_test[f'target_t{horizonte}']
            print("XGBoost: OK")
        except Exception as e:
            print(f"Error al cargar/predecir XGBoost t+{horizonte}: {e}")
        
        # Cargar y predecir con LightGBM
        try:
            print(f"\nCargando modelo LightGBM t+{horizonte}...")
            modelo_lgb = joblib.load(f'modelos/lightgbm/modelo_lightgbm_t{horizonte}.joblib')
            print("Modelo LightGBM cargado. Feature importances:")
            print(pd.Series(modelo_lgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head())
            
            X_test_lgb = df_test[feature_cols].fillna(method='ffill').fillna(method='bfill')
            predicciones['LightGBM'] = modelo_lgb.predict(X_test_lgb)
            print("\nEstadísticas de las predicciones LightGBM:")
            print(pd.Series(predicciones['LightGBM']).describe())
            
            errores['LightGBM'] = predicciones['LightGBM'] - df_test[f'target_t{horizonte}']
            print("LightGBM: OK")
        except Exception as e:
            print(f"Error al cargar/predecir LightGBM t+{horizonte}: {e}")
        
        # Cargar y predecir con SARIMA
        try:
            print(f"\nCargando modelo SARIMA t+{horizonte}...")
            modelo_sarima = joblib.load(f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
            
            # Para SARIMA, usamos solo la serie temporal objetivo
            serie_temporal = df_test[f'target_t{horizonte}'].copy()
            
            # Realizar predicciones directamente para todo el período
            predicciones_sarima = modelo_sarima.get_forecast(steps=len(serie_temporal))
            predicciones['SARIMA'] = predicciones_sarima.predicted_mean.values  # Convertir a array de numpy
            
            errores['SARIMA'] = predicciones['SARIMA'] - df_test[f'target_t{horizonte}'].values  # Usar .values para asegurar compatibilidad
            print("SARIMA: OK")
            
        except Exception as e:
            print(f"Error al cargar/predecir SARIMA t+{horizonte}: {e}")
            print("Tipo de error:", type(e))
            import traceback
            print(traceback.format_exc())
        
        # Generar tabla de predicciones
        print("\nGenerando tabla de predicciones...")
        generar_tabla_predicciones(df_test, predicciones, horizonte)
        
        # Generar gráficos
        print("Generando gráficos...")
        plot_predicciones_vs_real(df_test, predicciones, horizonte, 
                                f'graficos_comparacion/predicciones_vs_real_t{horizonte}.png')
        
        plot_distribucion_errores(errores, horizonte,
                                f'graficos_comparacion/distribucion_errores_t{horizonte}.png')
        
        # Calcular y guardar métricas
        print("\nCalculando métricas...")
        metricas = {}
        for modelo in predicciones.keys():
            metricas[modelo] = {
                'RMSE': np.sqrt(np.mean(errores[modelo]**2)),
                'MAE': np.mean(np.abs(errores[modelo])),
                'R2': 1 - np.sum(errores[modelo]**2) / np.sum((df_test[f'target_t{horizonte}'] - df_test[f'target_t{horizonte}'].mean())**2)
            }
            print(f"\nMétricas para {modelo}:")
            print(pd.Series(metricas[modelo]))
        
        # Guardar métricas para este horizonte
        metricas_por_horizonte[horizonte] = metricas
        
        # Guardar métricas en formato markdown
        with open(f'resultados/comparacion/metricas_t{horizonte}.md', 'w') as f:
            f.write(f"### Métricas de Error para Horizonte t+{horizonte}\n\n")
            f.write(pd.DataFrame(metricas).to_markdown())
    
    # Generar gráficos comparativos de métricas
    print("\nGenerando gráficos comparativos de métricas...")
    plot_metricas_comparativas(metricas_por_horizonte)
    
    print("\n¡Proceso completado!")
    print("Los resultados han sido guardados en:")
    print("- graficos_comparacion/")
    print("- resultados/comparacion/")

if __name__ == "__main__":
    main() 