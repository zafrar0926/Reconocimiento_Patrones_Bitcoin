import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def evaluar_modelo(y_true, y_pred, horizonte):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nMétricas para horizonte t+{horizonte}:")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_predicciones(y_true, y_pred, titulo, guardar_como=None):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.index, y_true.values, label='Real', alpha=0.8)
    plt.plot(y_true.index, y_pred, label='Predicción', alpha=0.8)
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel('Retorno Logarítmico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como)
    plt.close()

def ajustar_sarima_rapido(serie, horizonte):
    """Ajusta un modelo SARIMA con configuraciones predefinidas según el horizonte"""
    
    # Configuraciones optimizadas por horizonte
    configs = {
        1: ((1,1,1), (1,1,1,24)),  # Para predicciones a 1 hora
        6: ((2,1,1), (0,1,1,24)),  # Para predicciones a 6 horas
        12: ((1,1,2), (1,1,1,24))  # Para predicciones a 12 horas
    }
    
    order, seasonal_order = configs.get(horizonte, ((1,1,1), (0,1,1,24)))
    
    try:
        modelo = SARIMAX(
            serie,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        resultado = modelo.fit(disp=False, maxiter=50)
        return order, seasonal_order, resultado
    except Exception as e:
        print(f"Error con configuración inicial: {str(e)}")
        print("Intentando con modelo más simple...")
        # Si falla, intentar con un modelo más simple
        order = (1,1,1)
        seasonal_order = (0,1,1,24)
        try:
            modelo = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            resultado = modelo.fit(disp=False, maxiter=50)
            return order, seasonal_order, resultado
        except Exception as e:
            print(f"Error con modelo simple: {str(e)}")
            raise

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = [
        'modelos/sarima',
        'graficos/sarima',
        'resultados/sarima'
    ]
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

# Crear directorios
crear_directorios()

try:
    # === Cargar datos ===
    print("Cargando datos...")
    df_train = pd.read_csv('datos_procesados/features/btc_features_train.csv')
    df_test = pd.read_csv('datos_procesados/features/btc_features_test.csv')

    print(f"Shape de datos de entrenamiento: {df_train.shape}")
    print(f"Shape de datos de prueba: {df_test.shape}")

    for df in [df_train, df_test]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # === Ejecutar por horizonte ===
    horizontes = [1, 6, 12]
    resultados = {}

    for horizonte in horizontes:
        print(f"\n{'='*50}")
        print(f"🔍 Ajustando SARIMA para t+{horizonte}h")
        print(f"{'='*50}")
        
        try:
            # Concatenar series y tomar solo los últimos datos necesarios
            serie_total = pd.concat([
                df_train[f'target_t{horizonte}'],
                df_test[f'target_t{horizonte}']
            ]).dropna()
            
            # Asegurar que el índice temporal esté correctamente configurado
            serie_total.index = pd.DatetimeIndex(serie_total.index)
            serie_total = serie_total.asfreq('H')
            
            print(f"Total de datos disponibles: {len(serie_total)}")
            
            # Usar solo los últimos 2000 datos para entrenamiento
            n_train = min(2000, int(len(serie_total) * 0.8))
            serie_train = serie_total[-n_train:]
            serie_test = serie_total[-n_train:].tail(500)  # Usar últimos 500 datos para test
            
            # Asegurar que los índices estén correctamente configurados
            serie_train = serie_train.asfreq('H')
            serie_test = serie_test.asfreq('H')
            
            print(f"Datos de entrenamiento: {len(serie_train)}")
            print(f"Datos de prueba: {len(serie_test)}")
            print(f"Rango de fechas entrenamiento: {serie_train.index[0]} a {serie_train.index[-1]}")
            print(f"Rango de fechas prueba: {serie_test.index[0]} a {serie_test.index[-1]}")
            
            # Verificar valores NaN
            if serie_train.isna().any():
                print("⚠️ Advertencia: Hay valores NaN en los datos de entrenamiento")
                serie_train = serie_train.fillna(method='ffill').fillna(method='bfill')
            
            print("\nAjustando modelo SARIMA...")
            order, seasonal_order, resultado = ajustar_sarima_rapido(serie_train, horizonte)
            print(f"✅ Configuración: order={order}, seasonal_order={seasonal_order}")
            print(f"📉 AIC del modelo: {resultado.aic:.2f}")
            
            # Predicción en test
            pred = resultado.get_forecast(steps=len(serie_test))
            pred_mean = pred.predicted_mean
            y_true = serie_test[:len(pred_mean)]
            
            # Evaluación
            metricas = evaluar_modelo(y_true, pred_mean, horizonte)
            plot_predicciones(
                y_true, pred_mean,
                f'Predicción SARIMA para t+{horizonte}h',
                f'graficos/sarima/sarima_predicciones_t{horizonte}.png'
            )
            
            # Guardar resultados
            resultados[f't+{horizonte}'] = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': resultado.aic,
                'test_scores': metricas
            }
            
            # Guardar predicciones
            predicciones_df = pd.DataFrame({
                'timestamp': y_true.index,
                'real': y_true.values,
                'prediccion': pred_mean
            })
            predicciones_df.to_csv(f'resultados/sarima/predicciones_t{horizonte}.csv', index=False)
            print(f"Predicciones guardadas en 'resultados/sarima/predicciones_t{horizonte}.csv'")
            
            # Guardar modelo
            joblib.dump(resultado, f'modelos/sarima/modelo_sarima_t{horizonte}.joblib')
            
        except Exception as e:
            print(f"❌ Error procesando horizonte t+{horizonte}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    # Guardar resultados en CSV
    resultados_df = pd.DataFrame({
        horizonte: {
            'rmse': res['test_scores']['rmse'],
            'mae': res['test_scores']['mae'],
            'r2': res['test_scores']['r2']
        }
        for horizonte, res in resultados.items()
    })

    resultados_df.to_csv('resultados/sarima/resultados_sarima.csv')
    print("\n✅ Proceso completado exitosamente")
    print("📊 Resultados guardados en:")
    print("  - resultados/sarima/resultados_sarima.csv")
    print("  - graficos/sarima/*.png")
    print("  - modelos/sarima/modelo_sarima_t*.joblib")

except Exception as e:
    print(f"\n❌ Error general: {str(e)}")
    import traceback
    print(traceback.format_exc())
