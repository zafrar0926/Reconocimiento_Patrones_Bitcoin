import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def crear_directorios():
    directorios = [
        'modelos/xgboost',
        'graficos/xgboost',
        'resultados/xgboost'
    ]
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

def evaluar_modelo(y_true, y_pred, horizonte):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nMétricas para horizonte t+{horizonte}:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def penalized_rmse(y_true, y_pred):
    """RMSE con penalización por varianza y valores extremos"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Penalización por varianza excesiva
    std_pred = np.std(y_pred)
    std_true = np.std(y_true)
    var_penalty = abs(std_pred - std_true) / std_true
    
    # Penalización por valores extremos
    q1_true, q3_true = np.percentile(y_true, [25, 75])
    iqr_true = q3_true - q1_true
    outlier_mask = (y_pred < (q1_true - 1.5 * iqr_true)) | (y_pred > (q3_true + 1.5 * iqr_true))
    outlier_penalty = np.sum(outlier_mask) / len(y_true)
    
    return rmse * (1 + var_penalty + outlier_penalty)

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

def plot_importancia_caracteristicas(modelo, feature_cols, horizonte, guardar_como=None):
    importancia = pd.DataFrame({
        'caracteristica': feature_cols,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
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
    """Realizar búsqueda aleatoria de hiperparámetros con enfoque en regularización"""
    param_distributions = {
        'n_estimators': np.arange(100, 500, 50),
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [3, 4, 5, 6, 7],  # Reducido para evitar overfitting
        'min_child_weight': [1, 3, 5, 7],  # Aumentado para mejor regularización
        'subsample': [0.6, 0.7, 0.8, 0.9],  # Más variación en subsample
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.3],  # Regularización de split
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],  # L1 regularización
        'reg_lambda': [0.1, 1, 5, 10, 20],  # L2 regularización
        'scale_pos_weight': [1],  # Balance de clases si es necesario
    }
    
    # Configuración base del modelo
    model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        booster='gbtree',
        tree_method='hist'  # Más eficiente y con mejor regularización
    )
    
    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Búsqueda aleatoria de hiperparámetros
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,  # Aumentado para explorar más combinaciones
        cv=tscv,
        scoring=make_scorer(penalized_rmse, greater_is_better=False),
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nMejores parámetros encontrados:")
    print(random_search.best_params_)
    print("\nMejor puntaje:", -random_search.best_score_)
    
    return random_search.best_estimator_, random_search.best_params_

def plot_train_test_comparison(y_train, y_train_pred, y_test, y_test_pred, horizonte, guardar_como=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    ax1.plot(y_train.index, y_train.values, label='Real (Train)', alpha=0.8)
    ax1.plot(y_train.index, y_train_pred, label='Predicción (Train)', alpha=0.8)
    ax1.set_title(f'Comparación Train - Horizonte t+{horizonte}')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Retorno Logarítmico')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax2.plot(y_test.index, y_test.values, label='Real (Test)', alpha=0.8)
    ax2.plot(y_test.index, y_test_pred, label='Predicción (Test)', alpha=0.8)
    ax2.set_title(f'Comparación Test - Horizonte t+{horizonte}')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Retorno Logarítmico')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como)
    plt.close()

def main():
    crear_directorios()
    print("Cargando datos...")
    df_train = pd.read_csv('datos_procesados/features/btc_features_train.csv')
    df_test = pd.read_csv('datos_procesados/features/btc_features_test.csv')
    for df in [df_train, df_test]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    target_cols = ['target_t1', 'target_t6', 'target_t12']
    cols_excluir = ['timestamp'] + target_cols
    feature_cols = [col for col in df_train.columns if col not in cols_excluir and not col.startswith('log_return')]
    resultados, resultados_test, resultados_train, mejores_params = {}, {}, {}, {}
    for target in target_cols:
        horizonte = int(target.split('_t')[1])
        print(f"\nEntrenando modelo para horizonte t+{horizonte}")
        X_train = df_train[feature_cols]
        y_train = df_train[target]
        X_test = df_test[feature_cols]
        y_test = df_test[target]
        print("\nIniciando búsqueda de hiperparámetros...")
        modelo_final, mejores_params[target] = buscar_hiperparametros(X_train, y_train)
        print("\nEntrenando modelo final...")
        modelo_final.fit(X_train, y_train)
        y_train_pred = modelo_final.predict(X_train)
        metricas_train = evaluar_modelo(y_train, y_train_pred, horizonte)
        resultados_train[target] = metricas_train
        y_test_pred = modelo_final.predict(X_test)
        metricas_test = evaluar_modelo(y_test, y_test_pred, horizonte)
        resultados_test[target] = metricas_test
        pd.DataFrame({'timestamp': X_train.index, 'real': y_train.values, 'prediccion': y_train_pred}).to_csv(f'resultados/xgboost/predicciones_train_t{horizonte}.csv', index=False)
        pd.DataFrame({'timestamp': X_test.index, 'real': y_test.values, 'prediccion': y_test_pred}).to_csv(f'resultados/xgboost/predicciones_test_t{horizonte}.csv', index=False)
        plot_train_test_comparison(y_train, y_train_pred, y_test, y_test_pred, horizonte, f'graficos/xgboost/xgboost_train_test_comparison_t{horizonte}.png')
        joblib.dump(modelo_final, f'modelos/xgboost/modelo_xgboost_t{horizonte}.joblib')
        importancia = plot_importancia_caracteristicas(modelo_final, feature_cols, horizonte, f'graficos/xgboost/xgboost_importancia_t{horizonte}.png')
        resultados[target] = {
            'params': mejores_params[target],
            'train_scores': metricas_train,
            'test_scores': metricas_test,
            'feature_importance': importancia.to_dict()
        }
    print("\nGuardando resultados...")
    pd.DataFrame({
        'train': {k: v['train_scores'] for k, v in resultados.items()},
        'test': {k: v['test_scores'] for k, v in resultados.items()}
    }).to_csv('resultados/xgboost/resultados_comparacion_xgboost.csv')
    with open('resultados/xgboost/mejores_params_xgboost.txt', 'w') as f:
        f.write("Mejores parámetros por horizonte:\n\n")
        for target, params in mejores_params.items():
            f.write(f"Horizonte {target}:\n{params}\n\n")
    print("\n¡Proceso completado!")
    print("Resultados guardados en:")
    print("- resultados/xgboost/resultados_comparacion_xgboost.csv")
    print("- resultados/xgboost/predicciones_train_t*.csv")
    print("- resultados/xgboost/predicciones_test_t*.csv")
    print("- resultados/xgboost/mejores_params_xgboost.txt")
    print("- graficos/xgboost/*.png")

if __name__ == "__main__":
    main()
