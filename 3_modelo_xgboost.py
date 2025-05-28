import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = [
        'modelos/xgboost',
        'graficos/xgboost',
        'resultados/xgboost'
    ]
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

def evaluar_modelo(y_true, y_pred, horizonte):
    """Evaluar el modelo con múltiples métricas"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMétricas para horizonte t+{horizonte}:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_predicciones(y_true, y_pred, titulo, guardar_como=None):
    """Graficar predicciones vs valores reales"""
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

def plot_importancia_caracteristicas(modelo, feature_cols, horizonte, guardar_como=None):
    """Graficar importancia de características"""
    importancia = pd.DataFrame({
        'caracteristica': feature_cols,
        'importancia': modelo.feature_importances_
    })
    importancia = importancia.sort_values('importancia', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(importancia['caracteristica'][:10], importancia['importancia'][:10])
    plt.title(f'Top 10 Características Importantes para t+{horizonte}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como)
    plt.close()
    
    return importancia

def buscar_hiperparametros(X_train, y_train):
    """Realizar búsqueda aleatoria de hiperparámetros"""
    param_distributions = {
        'n_estimators': np.arange(100, 1500, 100),
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'max_depth': np.arange(3, 11),
        'min_child_weight': np.arange(1, 7),
        'subsample': np.linspace(0.6, 1.0, 20),
        'colsample_bytree': np.linspace(0.6, 1.0, 20),
        'gamma': np.linspace(0, 0.5, 20),
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
        'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0]
    }
    
    # Configuración base del modelo
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
    
    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Búsqueda aleatoria de hiperparámetros
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=500,  # Número de combinaciones a probar
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nMejores parámetros encontrados:")
    print(random_search.best_params_)
    print("\nMejor puntaje:", -random_search.best_score_)
    
    return random_search.best_params_

def main():
    # Crear directorios necesarios
    crear_directorios()
    
    # Cargar datos procesados
    print("Cargando datos...")
    df_train = pd.read_csv('datos_procesados/features/btc_features_train.csv')
    df_test = pd.read_csv('datos_procesados/features/btc_features_test.csv')

    # Convertir timestamp a datetime y establecer como índice
    for df in [df_train, df_test]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Definir columnas objetivo y características
    target_cols = ['target_t1', 'target_t6', 'target_t12']
    cols_excluir = ['timestamp'] + target_cols
    feature_cols = [col for col in df_train.columns if col not in cols_excluir and not col.startswith('log_return')]

    # Entrenamiento y evaluación para cada horizonte temporal
    resultados = {}
    resultados_test = {}
    mejores_params = {}

    for target in target_cols:
        horizonte = int(target.split('_t')[1])
        print(f"\nEntrenando modelo para horizonte t+{horizonte}")
        
        # Preparar datos
        X_train = df_train[feature_cols]
        y_train = df_train[target]
        X_test = df_test[feature_cols]
        y_test = df_test[target]
        
        # Búsqueda de hiperparámetros
        print("\nIniciando búsqueda de hiperparámetros...")
        mejores_params[target] = buscar_hiperparametros(X_train, y_train)
        
        # Entrenar modelo final con los mejores parámetros
        print("\nEntrenando modelo final...")
        modelo_final = XGBRegressor(**mejores_params[target], random_state=42)
        modelo_final.fit(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        y_pred_test = modelo_final.predict(X_test)
        metricas_test = evaluar_modelo(y_test, y_pred_test, horizonte)
        resultados_test[target] = metricas_test

        # Guardar el modelo entrenado
        joblib.dump(modelo_final, f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
        print(f"Modelo guardado como 'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib'")
        
        # Graficar predicciones
        plot_predicciones(
            y_test, y_pred_test,
            f'Predicciones XGBoost para t+{horizonte} (Test)',
            f'graficos/xgboost/xgboost_predicciones_t{horizonte}.png'
        )
        
        # Graficar importancia de características
        importancia = plot_importancia_caracteristicas(
            modelo_final, feature_cols, horizonte,
            f'graficos/xgboost/xgboost_importancia_t{horizonte}.png'
        )
        
        # Guardar resultados
        resultados[target] = {
            'params': mejores_params[target],
            'test_scores': metricas_test,
            'feature_importance': importancia.to_dict()
        }

    # Guardar resultados
    print("\nGuardando resultados...")
    resultados_df = pd.DataFrame({k: v['test_scores'] for k, v in resultados.items()}).T
    resultados_df.to_csv('resultados/xgboost/resultados_xgboost.csv')

    # Guardar mejores parámetros
    with open('resultados/xgboost/mejores_params_xgboost.txt', 'w') as f:
        f.write("Mejores parámetros por horizonte:\n\n")
        for target, params in mejores_params.items():
            f.write(f"Horizonte {target}:\n")
            f.write(str(params))
            f.write("\n\n")

    print("\n¡Proceso completado!")
    print("Resultados guardados en:")
    print("- resultados/xgboost/resultados_xgboost.csv")
    print("- resultados/xgboost/mejores_params_xgboost.txt")
    print("- graficos/xgboost/*.png")

if __name__ == "__main__":
    main() 